import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TRAIN_FILE = "data/WebQSP/generation/merged/WebQSP_train.json"
TEST_FILE  = "data/WebQSP/generation/merged/WebQSP_test.json"

EMBED_MODEL = "text-embedding-3-large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 50
LR = 2e-4

CACHE_DIR = Path("ragnet/embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

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

    print(f"{path}: {len(data)} examples")
    return data


# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

def embed_texts(texts, cache_file):

    if cache_file.exists():
        print("Loading", cache_file)
        return np.load(cache_file)

    vecs = []

    for i in tqdm(range(0, len(texts), 32)):
        batch = texts[i:i+32]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        vecs.extend([x.embedding for x in resp.data])

    vecs = np.array(vecs, dtype=np.float32)
    np.save(cache_file, vecs)
    return vecs


def normalize(x):
    return F.normalize(torch.tensor(x).to(DEVICE), dim=-1)


# ------------------------------------------------
# JOINT EMBEDDING
# ------------------------------------------------

def build_joint(q, s):
    """
    Joint representation of Question + Logical Form
    """
    return F.normalize(q + s, dim=-1)


# ------------------------------------------------
# SMALL ADAPTER
# ------------------------------------------------

class SmallAdapter(nn.Module):

    def __init__(self, dim=3072, hidden=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.normalize(x + self.scale * self.net(x), dim=-1)


# ------------------------------------------------
# LOSS
# ------------------------------------------------

def contrastive_loss(zq, zs, temperature=0.05):

    logits = zq @ zs.T / temperature
    labels = torch.arange(len(zq), device=zq.device)

    loss_q = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)

    return (loss_q + loss_s) / 2


# ------------------------------------------------
# RETRIEVAL
# ------------------------------------------------

@torch.no_grad()
def nearest_neighbor(query, database):
    sim = query @ database.T
    return torch.argmax(sim, dim=1)


# ------------------------------------------------
# AGREEMENT EVAL
# ------------------------------------------------

@torch.no_grad()
def evaluate_alignment(
    test_q,
    test_s,
    test_joint,
    train_q,
    train_s,
    train_joint,
    label,
):

    print(f"\n===== {label} =====")

    # GOLD STANDARD = joint semantic pair
    gold = nearest_neighbor(test_joint, train_joint)

    # Variants
    qq = nearest_neighbor(test_q, train_q)
    qs = nearest_neighbor(test_q, train_s)
    qj = nearest_neighbor(test_q, train_joint)

    total = len(gold)

    qq_agree = (qq == gold).sum().item()
    qs_agree = (qs == gold).sum().item()
    qj_agree = (qj == gold).sum().item()

    print(f"Q→Q agreement:   {qq_agree/total:.4f} ({qq_agree}/{total})")
    print(f"Q→S agreement:   {qs_agree/total:.4f} ({qs_agree}/{total})")
    print(f"Q→Q+S agreement: {qj_agree/total:.4f} ({qj_agree}/{total})")


# ------------------------------------------------
# TRAIN ADAPTER
# ------------------------------------------------

def train_adapter(train_q, train_s):

    adapter_q = SmallAdapter().to(DEVICE)
    adapter_s = SmallAdapter().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(adapter_q.parameters()) +
        list(adapter_s.parameters()),
        lr=LR,
    )

    N = train_q.size(0)

    print("\nTraining small adapter...")

    for epoch in range(EPOCHS):

        perm = torch.randperm(N)

        total_loss = 0

        for i in range(0, N, BATCH_SIZE):

            idx = perm[i:i+BATCH_SIZE]

            q = train_q[idx]
            s = train_s[idx]

            zq = adapter_q(q)
            zs = adapter_s(s)

            loss = contrastive_loss(zq, zs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"loss={total_loss/(N//BATCH_SIZE):.4f}"
        )

    return adapter_q, adapter_s


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    train = load_data(TRAIN_FILE)
    test  = load_data(TEST_FILE)

    train_q_text = [x[0] for x in train]
    train_s_text = [x[1] for x in train]

    test_q_text = [x[0] for x in test]
    test_s_text = [x[1] for x in test]

    # ---------- embeddings ----------
    train_q = normalize(embed_texts(
        train_q_text,
        CACHE_DIR / "train_q.npy"
    ))

    train_s = normalize(embed_texts(
        train_s_text,
        CACHE_DIR / "train_s.npy"
    ))

    test_q = normalize(embed_texts(
        test_q_text,
        CACHE_DIR / "test_q.npy"
    ))

    test_s = normalize(embed_texts(
        test_s_text,
        CACHE_DIR / "test_s.npy"
    ))

    # ------------------------------------------------
    # BASELINE
    # ------------------------------------------------

    train_joint = build_joint(train_q, train_s)
    test_joint  = build_joint(test_q, test_s)

    evaluate_alignment(
        test_q,
        test_s,
        test_joint,
        train_q,
        train_s,
        train_joint,
        "BASELINE EMBEDDINGS"
    )

    # ------------------------------------------------
    # TRAIN ADAPTER
    # ------------------------------------------------

    adapter_q, adapter_s = train_adapter(train_q, train_s)

    # ------------------------------------------------
    # ADAPTED EMBEDDINGS
    # ------------------------------------------------

    with torch.no_grad():
        train_q_a = adapter_q(train_q)
        train_s_a = adapter_s(train_s)

        test_q_a = adapter_q(test_q)
        test_s_a = adapter_s(test_s)

        train_joint_a = build_joint(train_q_a, train_s_a)
        test_joint_a  = build_joint(test_q_a, test_s_a)

    evaluate_alignment(
        test_q_a,
        test_s_a,
        test_joint_a,
        train_q_a,
        train_s_a,
        train_joint_a,
        "AFTER SMALL ADAPTER"
    )


# ------------------------------------------------

if __name__ == "__main__":
    main()
