import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TRAIN_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_train.json"
TEST_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"

EMBED_MODEL = "text-embedding-3-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4

EMB_DIM = 3072

CACHE_DIR = Path("ragnet/embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

Q_EMB_FILE = CACHE_DIR / "question_embs.npy"
S_EMB_FILE = CACHE_DIR / "sexpr_embs.npy"

TEST_Q_EMB_FILE = CACHE_DIR / "test_question_embs.npy"
TEST_S_EMB_FILE = CACHE_DIR / "test_sexpr_embs.npy"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------------------------------------------
# DATA
# ------------------------------------------------

def load_data(path):
    with open(path) as f:
        raw = json.load(f)

    data = []
    for ex in raw:
        q = ex.get("question", "").strip()
        s = ex.get("normed_sexpr", "").strip()
        if q and s:
            data.append((q, s))

    print(f"Loaded {len(data)} examples")
    return data


# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

def embed_texts(texts, batch_size=32):

    all_vecs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        all_vecs.extend([x.embedding for x in resp.data])

    return np.array(all_vecs, dtype=np.float32)


def load_or_create_embeddings(data, q_file, s_file):

    if q_file.exists() and s_file.exists():
        print("Loading cached embeddings...")
        return (
            np.load(q_file).astype(np.float32),
            np.load(s_file).astype(np.float32),
        )

    questions = [x[0] for x in data]
    sexprs = [x[1] for x in data]

    q_embs = embed_texts(questions)
    s_embs = embed_texts(sexprs)

    np.save(q_file, q_embs)
    np.save(s_file, s_embs)

    return q_embs, s_embs


# ------------------------------------------------
# DATASET
# ------------------------------------------------

class PairDataset(Dataset):

    def __init__(self, q_embs, s_embs):
        self.q = torch.tensor(q_embs, dtype=torch.float32)
        self.s = torch.tensor(s_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return self.q[i], self.s[i]


# ------------------------------------------------
# MODEL BLOCKS
# ------------------------------------------------

class LoRAExpert(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.B(self.A(x))


class GeometryBlock(nn.Module):
    def __init__(self, dim, rank=32, num_experts=2):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.experts = nn.ModuleList([
            LoRAExpert(dim, rank)
            for _ in range(num_experts)
        ])

        self.gate = nn.Parameter(torch.zeros(num_experts))
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.norm(x)

        weights = torch.softmax(self.gate, dim=0)

        update = 0
        for w, expert in zip(weights, self.experts):
            update = update + w * expert(h)

        return x + torch.tanh(self.res_scale) * update


# ------------------------------------------------
# ADAPTER (SMALL + STABLE)
# ------------------------------------------------

class SOTAEmbeddingAdapter(nn.Module):

    def __init__(self, dim=3072, bottleneck=512, depth=2, num_vectors=4):
        super().__init__()

        self.input_norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(dim))

        self.down = nn.Linear(dim, bottleneck)

        self.blocks = nn.ModuleList([
            GeometryBlock(bottleneck)
            for _ in range(depth)
        ])

        self.mid_norm = nn.LayerNorm(bottleneck)
        self.up = nn.Linear(bottleneck, dim)

        self.multi_head = nn.Linear(dim, dim * num_vectors)
        self.vector_weights = nn.Parameter(torch.ones(num_vectors))

        # CLIP-style temperature
        self.logit_scale = nn.Parameter(torch.tensor(4.6))

    def forward(self, x):

        residual = x

        x = self.input_norm(x)
        x = x * self.scale
        x = self.down(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.mid_norm(x)
        x = self.up(x)

        x = residual + x

        x = self.multi_head(x)

        B = x.size(0)
        x = x.view(B, -1, residual.size(-1))

        weights = torch.softmax(self.vector_weights, dim=0)

        return x, weights, self.logit_scale.exp()


# ------------------------------------------------
# BASELINE
# ------------------------------------------------

class IdentityModel(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(1)
        return x, torch.tensor([1.0], device=x.device), torch.tensor(1.0, device=x.device)


# ------------------------------------------------
# MULTI VECTOR COLLAPSE
# ------------------------------------------------

def collapse_vectors(z, weights):

    z = torch.sum(z * weights.view(1, -1, 1), dim=1)
    return F.normalize(z, dim=-1)


# ------------------------------------------------
# BIDIRECTIONAL INFONCE
# ------------------------------------------------

def contrastive_loss(zq, zs, temperature):

    logits = (zq @ zs.T) / temperature

    labels = torch.arange(len(zq), device=zq.device)

    loss_q = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)

    return (loss_q + loss_s) / 2


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

@torch.no_grad()
def evaluate(q_model, s_model, label):

    print(f"\nRunning evaluation: {label}")

    data = load_data(TEST_GENERATION_DATA_NAME)

    q_embs, s_embs = load_or_create_embeddings(
        data,
        TEST_Q_EMB_FILE,
        TEST_S_EMB_FILE
    )

    q = torch.tensor(q_embs).to(DEVICE)
    s = torch.tensor(s_embs).to(DEVICE)

    q_model.eval()
    s_model.eval()

    zq, wq, _ = q_model(q)
    zs, ws, _ = s_model(s)

    zq = collapse_vectors(zq, wq)
    zs = collapse_vectors(zs, ws)

    sim = zq @ zs.T

    total = len(zq)

    for k in [1,2,3,4,5]:
        correct = 0
        for i in range(total):
            if i in torch.topk(sim[i], k).indices:
                correct += 1

        print(f"Top-{k} Accuracy: {correct/total:.4f} ({correct}/{total})")


# ------------------------------------------------
# TRAIN
# ------------------------------------------------

def main():

    train_data = load_data(TRAIN_GENERATION_DATA_NAME)

    q_embs, s_embs = load_or_create_embeddings(
        train_data,
        Q_EMB_FILE,
        S_EMB_FILE
    )

    dataset = PairDataset(q_embs, s_embs)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    q_model = SOTAEmbeddingAdapter().to(DEVICE)
    s_model = SOTAEmbeddingAdapter().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(q_model.parameters()) +
        list(s_model.parameters()),
        lr=LR,
        weight_decay=1e-2
    )

    baseline_q = IdentityModel().to(DEVICE)
    baseline_s = IdentityModel().to(DEVICE)

    evaluate(baseline_q, baseline_s, "BASELINE")

    print("\nTraining adapter...")

    for epoch in range(EPOCHS):

        q_model.train()
        s_model.train()

        total_loss = 0

        for q, s in tqdm(loader):

            q = q.to(DEVICE)
            s = s.to(DEVICE)

            zq, wq, logit_scale = q_model(q)
            zs, ws, _ = s_model(s)

            zq = collapse_vectors(zq, wq)
            zs = collapse_vectors(zs, ws)

            temperature = 1 / logit_scale

            loss = contrastive_loss(zq, zs, temperature)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(q_model.parameters()) +
                list(s_model.parameters()),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(loader):.4f}")

        evaluate(q_model, s_model, f"ADAPTER Epoch {epoch+1}")

    torch.save({
        "q_adapter": q_model.state_dict(),
        "s_adapter": s_model.state_dict()
    }, "adapter.pt")

    print("Saved adapter.pt")


# ------------------------------------------------

if __name__ == "__main__":
    main()