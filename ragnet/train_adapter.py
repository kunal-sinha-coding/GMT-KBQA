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
EPOCHS = 10
LR = 1e-4
PROJ_DIM = 768
TOP_K = 3

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

    examples = []
    for ex in raw:
        q = ex.get("question", "").strip()
        s = ex.get("normed_sexpr", "").strip()

        if not q or not s:
            continue

        examples.append({
            "question": q,
            "normed_sexpr": s
        })

    print(f"Loaded {len(examples)} pairs from {path}")
    return examples


# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

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
# ADAPTER
# ------------------------------------------------

class Adapter(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ------------------------------------------------
# LOSS (Bidirectional Contrastive)
# ------------------------------------------------

def contrastive_loss(q, s, temperature=0.07):

    sim = torch.matmul(q, s.T) / temperature
    labels = torch.arange(len(q)).to(q.device)

    loss_q = F.cross_entropy(sim, labels)
    loss_s = F.cross_entropy(sim.T, labels)

    return (loss_q + loss_s) / 2


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

@torch.no_grad()
def evaluate(model):

    print("\nRunning evaluation...")

    test_data = load_data(TEST_GENERATION_DATA_NAME)

    q_embs, s_embs = load_or_create_embeddings(
        test_data,
        TEST_Q_EMB_FILE,
        TEST_S_EMB_FILE
    )

    q_embs = torch.tensor(q_embs, dtype=torch.float32).to(DEVICE)
    s_embs = torch.tensor(s_embs, dtype=torch.float32).to(DEVICE)

    model.eval()

    zq = model(q_embs)
    zs = model(s_embs)

    sim = torch.matmul(zq, zs.T)

    correct = 0
    total = len(zq)

    for i in range(total):
        topk = torch.topk(sim[i], k=TOP_K).indices
        if i in topk:
            correct += 1

    acc = correct / total
    print(f"Top-{TOP_K} Accuracy: {acc:.4f} ({correct}/{total})")

    return acc


# ------------------------------------------------
# TRAIN
# ------------------------------------------------

def main():

    # ----- TRAIN DATA -----
    train_data = load_data(TRAIN_GENERATION_DATA_NAME)

    q_embs, s_embs = load_or_create_embeddings(
        train_data,
        Q_EMB_FILE,
        S_EMB_FILE
    )

    dataset = PairDataset(q_embs, s_embs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Adapter(
        in_dim=q_embs.shape[1],
        out_dim=PROJ_DIM
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Training adapter...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for q, s in tqdm(loader):

            q = q.to(DEVICE)
            s = s.to(DEVICE)

            zq = model(q)
            zs = model(s)

            loss = contrastive_loss(zq, zs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg:.4f}")

    torch.save(model.state_dict(), "adapter.pt")
    print("Saved adapter.pt")

    # ----- EVALUATE -----
    evaluate(model)


if __name__ == "__main__":
    main()