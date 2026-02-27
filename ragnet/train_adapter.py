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


# =====================================================
# CONFIG
# =====================================================

TRAIN_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_train.json"
TEST_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"

EMBED_MODEL = "text-embedding-3-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-4                # Increased again
TEMPERATURE = 0.02

EMB_DIM = 3072
QUEUE_SIZE = 8192
MOMENTUM = 0.99         # Less frozen than 0.995

CACHE_DIR = Path("ragnet/embedding_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

Q_EMB_FILE = CACHE_DIR / "question_embs.npy"
S_EMB_FILE = CACHE_DIR / "sexpr_embs.npy"

TEST_Q_EMB_FILE = CACHE_DIR / "test_question_embs.npy"
TEST_S_EMB_FILE = CACHE_DIR / "test_sexpr_embs.npy"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# =====================================================
# DATA
# =====================================================

def load_data(path):
    with open(path) as f:
        raw = json.load(f)

    examples = []
    for ex in raw:
        q = ex.get("question", "").strip()
        s = ex.get("normed_sexpr", "").strip()

        if q and s:
            examples.append({"question": q, "normed_sexpr": s})

    print(f"Loaded {len(examples)} examples from {path}")
    return examples


# =====================================================
# EMBEDDINGS
# =====================================================

def embed_texts(texts, batch_size=32):

    all_vecs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        all_vecs.extend([x.embedding for x in resp.data])

    return np.array(all_vecs)


def load_or_create_embeddings(data, q_file, s_file):

    if q_file.exists() and s_file.exists():
        print("Loading cached embeddings...")
        return np.load(q_file), np.load(s_file)

    questions = [x["question"] for x in data]
    sexprs = [x["normed_sexpr"] for x in data]

    q_embs = embed_texts(questions)
    s_embs = embed_texts(sexprs)

    np.save(q_file, q_embs)
    np.save(s_file, s_embs)

    return q_embs, s_embs


# =====================================================
# DATASET
# =====================================================

class PairDataset(Dataset):

    def __init__(self, q_embs, s_embs):
        self.q = torch.tensor(q_embs, dtype=torch.float32)
        self.s = torch.tensor(s_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return self.q[idx], self.s[idx]


# =====================================================
# RESIDUAL ADAPTER (STRONGER)
# =====================================================

class ResidualAdapter(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Stronger initial scale
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        delta = self.net(x)
        return F.normalize(x + self.scale * delta, dim=-1)


# =====================================================
# MOMENTUM UPDATE
# =====================================================

@torch.no_grad()
def momentum_update(model, momentum_model, m=MOMENTUM):
    for p, mp in zip(model.parameters(), momentum_model.parameters()):
        mp.data = mp.data * m + p.data * (1 - m)


# =====================================================
# MEMORY QUEUE
# =====================================================

class MemoryQueue:

    def __init__(self, dim, size, device):
        self.size = size
        self.device = device

        self.q_queue = F.normalize(
            torch.randn(size, dim, device=device), dim=-1
        )
        self.s_queue = F.normalize(
            torch.randn(size, dim, device=device), dim=-1
        )

        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, q, s):

        batch_size = q.size(0)
        end = self.ptr + batch_size

        if end <= self.size:
            self.q_queue[self.ptr:end] = q
            self.s_queue[self.ptr:end] = s
        else:
            first = self.size - self.ptr
            self.q_queue[self.ptr:] = q[:first]
            self.s_queue[self.ptr:] = s[:first]
            self.q_queue[:batch_size-first] = q[first:]
            self.s_queue[:batch_size-first] = s[first:]

        self.ptr = (self.ptr + batch_size) % self.size


# =====================================================
# CONTRASTIVE LOSS
# =====================================================

def contrastive_loss(q, s, queue):

    all_s = torch.cat([s, queue.s_queue.detach()], dim=0)
    all_q = torch.cat([q, queue.q_queue.detach()], dim=0)

    sim_q = torch.matmul(q, all_s.T) / TEMPERATURE
    sim_s = torch.matmul(s, all_q.T) / TEMPERATURE

    labels = torch.arange(len(q), device=q.device)

    loss_q = F.cross_entropy(sim_q, labels)
    loss_s = F.cross_entropy(sim_s, labels)

    return (loss_q + loss_s) / 2


# =====================================================
# EVALUATION
# =====================================================

@torch.no_grad()
def evaluate(q_model, s_model, label="MODEL"):

    print(f"\nRunning evaluation: {label}")

    test_data = load_data(TEST_GENERATION_DATA_NAME)

    q_embs, s_embs = load_or_create_embeddings(
        test_data,
        TEST_Q_EMB_FILE,
        TEST_S_EMB_FILE
    )

    q_embs = torch.tensor(q_embs, dtype=torch.float32).to(DEVICE)
    s_embs = torch.tensor(s_embs, dtype=torch.float32).to(DEVICE)

    q_model.eval()
    s_model.eval()

    zq = q_model(q_embs)
    zs = s_model(s_embs)

    sim = torch.matmul(zq, zs.T) / TEMPERATURE

    total = len(zq)
    ks = [1,2,3,4,5]
    correct = [0]*len(ks)

    for i in range(total):
        for j,k in enumerate(ks):
            if i in torch.topk(sim[i], k=k).indices:
                correct[j]+=1

    for j,k in enumerate(ks):
        print(f"{label} Top-{k}: {correct[j]/total:.4f}")


# =====================================================
# TRAIN
# =====================================================

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

    # Baseline evaluation (important)
    evaluate(
        lambda x: F.normalize(x, dim=-1),
        lambda x: F.normalize(x, dim=-1),
        label="BASELINE"
    )

    q_model = ResidualAdapter(EMB_DIM).to(DEVICE)
    s_model = ResidualAdapter(EMB_DIM).to(DEVICE)

    q_momentum = ResidualAdapter(EMB_DIM).to(DEVICE)
    s_momentum = ResidualAdapter(EMB_DIM).to(DEVICE)

    q_momentum.load_state_dict(q_model.state_dict())
    s_momentum.load_state_dict(s_model.state_dict())

    for p in q_momentum.parameters():
        p.requires_grad = False
    for p in s_momentum.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(q_model.parameters()) +
        list(s_model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    queue = MemoryQueue(EMB_DIM, QUEUE_SIZE, DEVICE)

    print("Warming memory queue...")
    with torch.no_grad():
        for q, s in loader:
            q = q.to(DEVICE)
            s = s.to(DEVICE)

            queue.enqueue(
                q_momentum(q),
                s_momentum(s)
            )

            if queue.ptr == 0:
                break

    print("Training...")

    for epoch in range(EPOCHS):

        q_model.train()
        s_model.train()

        total_loss = 0

        for q, s in tqdm(loader):

            q = q.to(DEVICE)
            s = s.to(DEVICE)

            zq = q_model(q)
            zs = s_model(s)

            loss = contrastive_loss(zq, zs, queue)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(q_model.parameters()) +
                list(s_model.parameters()),
                1.0
            )

            optimizer.step()

            with torch.no_grad():
                momentum_update(q_model, q_momentum)
                momentum_update(s_model, s_momentum)

                queue.enqueue(
                    q_momentum(q),
                    s_momentum(s)
                )

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(loader):.4f}")

    torch.save({
        "q_model": q_model.state_dict(),
        "s_model": s_model.state_dict()
    }, "adapter.pt")

    print("Saved adapter.pt")

    evaluate(q_model, s_model, label="ADAPTER")


if __name__ == "__main__":
    main()