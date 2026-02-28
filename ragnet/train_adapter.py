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

HARD_NEG_K = 20
HARD_NEG_PER_BATCH = 20#4
TEMPERATURE = 0.2

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

    # ✅ force float32
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
# HARD NEGATIVE MINING
# ------------------------------------------------

def build_hard_negative_index(q_embs, s_embs):

    print("Building hard negatives...")

    q = F.normalize(torch.tensor(q_embs, dtype=torch.float32), dim=-1)
    s = F.normalize(torch.tensor(s_embs, dtype=torch.float32), dim=-1)

    sim = q @ s.T

    hard_negs = []

    for i in tqdm(range(len(q))):
        sims = sim[i].clone()
        sims[i] = -1e9
        topk = torch.topk(sims, HARD_NEG_K).indices
        hard_negs.append(topk.tolist())

    return hard_negs

# ------------------------------------------------
# DATASET
# ------------------------------------------------

class PairDataset(Dataset):

    def __init__(self, q_embs, s_embs, hard_negs):

        self.q = torch.tensor(q_embs, dtype=torch.float32)
        self.s = torch.tensor(s_embs, dtype=torch.float32)
        self.hard_negs = hard_negs

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):

        neg_ids = np.random.choice(
            self.hard_negs[i],
            HARD_NEG_PER_BATCH,
            replace=False
        )

        neg_s = self.s[neg_ids]

        return self.q[i], self.s[i], neg_s


def collate_fn(batch):

    q = torch.stack([b[0] for b in batch])
    pos = torch.stack([b[1] for b in batch])
    neg = torch.cat([b[2] for b in batch], dim=0)

    return q, pos, neg

# ------------------------------------------------
# MODELS
# ------------------------------------------------

class ResidualAdapter(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return F.normalize(x + self.alpha * self.mlp(x), dim=-1)

# ------------------------------------------------
# Multi-Expert LoRA Geometry Block
# ------------------------------------------------
class LoRAExpert(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.B(self.A(x))


class GeometryBlock(nn.Module):
    def __init__(self, dim, rank=128, num_experts=2):
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
# SOTA Embedding Adapter
# ------------------------------------------------
class SOTAEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        dim=3072,
        bottleneck=1024,
        depth=8,
        rank=128,
        num_vectors=4,
    ):
        super().__init__()

        # ---------- Whitening / anisotropy fix ----------
        self.input_norm = nn.LayerNorm(dim)

        # ---------- Feature importance ----------
        self.scale = nn.Parameter(torch.ones(dim))

        # ---------- Manifold projection ----------
        self.down = nn.Linear(dim, bottleneck)

        # ---------- Deep geometry mixer ----------
        self.blocks = nn.ModuleList([
            GeometryBlock(bottleneck, rank)
            for _ in range(depth)
        ])

        self.mid_norm = nn.LayerNorm(bottleneck)

        # ---------- Return to embedding space ----------
        self.up = nn.Linear(bottleneck, dim)

        # ---------- Multi-vector head ----------
        self.multi_head = nn.Linear(dim, dim * num_vectors)
        self.vector_weights = nn.Parameter(torch.ones(num_vectors))

        # ---------- Learned similarity temperature ----------
        self.logit_scale = nn.Parameter(torch.tensor(4.6))

    def forward(self, x):
        residual = x

        # whitening
        x = self.input_norm(x)

        # feature reweighting
        x = x * self.scale

        # manifold projection
        x = self.down(x)

        # deep geometric adaptation
        for blk in self.blocks:
            x = blk(x)

        x = self.mid_norm(x)

        # back to embedding space
        x = self.up(x)

        # residual merge
        x = residual + x

        # multi-vector representation
        x = self.multi_head(x)
        B = x.size(0)
        x = x.view(B, -1, residual.size(-1))

        # normalize vectors
        x = F.normalize(x, dim=-1)

        weights = torch.softmax(self.vector_weights, dim=0)

        return x, weights, self.logit_scale.exp()


class IdentityModel(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

# ------------------------------------------------
# LOSS
# ------------------------------------------------

def contrastive_loss(q, pos_s, neg_s):

    pos_sim = torch.sum(q * pos_s, dim=-1, keepdim=True)
    neg_sim = q @ neg_s.T

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    logits /= TEMPERATURE

    labels = torch.zeros(len(q), dtype=torch.long, device=q.device)

    return F.cross_entropy(logits, labels)

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

    q = torch.tensor(q_embs, dtype=torch.float32).to(DEVICE)
    s = torch.tensor(s_embs, dtype=torch.float32).to(DEVICE)

    q_model.eval()
    s_model.eval()

    zq = q_model(q)
    zs = s_model(s)

    sim = zq @ zs.T

    total = len(zq)

    for k in [1,2,3,4,5]:
        correct = 0
        for i in range(total):
            topk = torch.topk(sim[i], k).indices
            if i in topk:
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

    hard_negs = build_hard_negative_index(q_embs, s_embs)

    dataset = PairDataset(q_embs, s_embs, hard_negs)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    q_model = SOTAEmbeddingAdapter(EMB_DIM).to(DEVICE)
    s_model = SOTAEmbeddingAdapter(EMB_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(q_model.parameters()) +
        list(s_model.parameters()),
        lr=LR
    )

    # ----- BASELINE -----
    baseline_q = IdentityModel().to(DEVICE)
    baseline_s = IdentityModel().to(DEVICE)

    evaluate(baseline_q, baseline_s, "BASELINE")

    # ----- TRAIN -----
    print("\nTraining adapter...")

    for epoch in range(EPOCHS):

        q_model.train()
        s_model.train()

        total_loss = 0

        for q, pos_s, neg_s in tqdm(loader):

            q = q.to(DEVICE)
            pos_s = pos_s.to(DEVICE)
            neg_s = neg_s.to(DEVICE)

            zq = q_model(q)
            zpos = s_model(pos_s)
            zneg = s_model(neg_s)

            loss = contrastive_loss(zq, zpos, zneg)

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
