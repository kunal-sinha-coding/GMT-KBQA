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

BATCH_SIZE = 2990
EPOCHS = 500
LR = 1e-3

CACHE_DIR = Path("ragnet/embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ADAPTER_PATH = Path("ragnet/adapter.pt")

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

    def __init__(self, dim=3072, hidden=8192):#512):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        #self.scale = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return F.normalize(x + self.scale * self.net(x), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden=2048):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)

        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        delta = self.fc2(h)

        gate = torch.sigmoid(self.gate(x))

        return x + self.scale * gate * delta


class SmallAdapterStacked(nn.Module):
    def __init__(self, dim=3072):
        super().__init__()

        self.blocks = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim),
            ResidualBlock(dim),
        )

    def forward(self, x):
        x = self.blocks(x)
        return F.normalize(x, dim=-1)

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
# HARD NEGATIVE CONTRASTIVE LOSS
# ------------------------------------------------

def contrastive_loss_hard(zq, zs, temperature=0.05, hard_k=16):

    """
    Hard-negative bidirectional contrastive loss.
    """

    sim = zq @ zs.T                      # [B,B]
    sim = sim / temperature

    B = sim.size(0)
    labels = torch.arange(B, device=sim.device)

    # -------------------------------
    # mask positives
    # -------------------------------
    mask = torch.eye(B, device=sim.device).bool()
    sim_neg = sim.masked_fill(mask, -1e9)

    # -------------------------------
    # mine hard negatives
    # -------------------------------
    hard_vals_q, hard_idx_q = torch.topk(
        sim_neg, k=min(hard_k, B-1), dim=1
    )

    hard_vals_s, hard_idx_s = torch.topk(
        sim_neg.T, k=min(hard_k, B-1), dim=1
    )

    # positives
    pos = sim[torch.arange(B), labels].unsqueeze(1)

    # build logits
    logits_q = torch.cat([pos, hard_vals_q], dim=1)
    logits_s = torch.cat([pos, hard_vals_s], dim=1)

    target = torch.zeros(B, dtype=torch.long, device=sim.device)

    loss_q = F.cross_entropy(logits_q, target)
    loss_s = F.cross_entropy(logits_s, target)

    return (loss_q + loss_s) / 2

def geometry_loss(z, base):
    """
    Preserve pairwise cosine similarities.
    """
    sim_new = z @ z.T
    sim_old = base @ base.T

    return F.mse_loss(sim_new, sim_old)

def contrastive_and_geometry_loss(zq, zs):
    align = contrastive_loss_hard(zq, zs)

    geom_q = geometry_loss(zq, zq)
    geom_s = geometry_loss(zs, zs)

    loss = align + 0.1 * (geom_q + geom_s)
    return loss

def kl_divergence_loss(
    student_logits,      # (B, D) adapter output
    teacher_logits,      # (B, D) joint embeddings (same batch)
    temperature=0.1
):
    """
    Distill retrieval behavior from joint embeddings.

    student_emb : adapter(E(q))
    teacher_emb : E(q + logical_form)
    train_joint : retrieval index (frozen)
    """

    # -----------------------------
    # Teacher retrieval distribution
    # -----------------------------
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # -----------------------------
    # Student retrieval distribution
    # -----------------------------
    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=-1
    )

    # -----------------------------
    # KL divergence
    # -----------------------------
    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    return loss

def topk_kl_divergence_loss(student_logits, teacher_logits, k=64, T=0.1):

    with torch.no_grad():
        topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)

        teacher_probs = F.softmax(topk_vals / T, dim=-1)

    student_topk = torch.gather(student_logits, 1, topk_idx)

    student_log_probs = F.log_softmax(student_topk / T, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

@torch.no_grad()
def compute_teacher_targets(train_joint, batch_size=256):
    N = train_joint.size(0)
    targets = []

    for i in range(0, N, batch_size):
        z = train_joint[i:i+batch_size]              # teacher query
        sims = z @ train_joint.T                     # similarity matrix

        # remove self-match
        row_ids = torch.arange(i, i+z.size(0), device=sims.device)
        sims[torch.arange(z.size(0)), row_ids] = -1e9

        top1 = sims.argmax(dim=1)
        targets.append(top1)

    return torch.cat(targets)


def retrieval_distill_loss(zq, train_joint, teacher_idx, temperature=0.1):
    logits = (zq @ train_joint.T) / temperature
    return F.cross_entropy(logits, teacher_idx)


def residual_difference_loss(zq, target_joint, original_q):
    """
    zq: The output of adapter(original_q) -> already normalized
    target_joint: The E(Q+S) from your teacher -> already normalized
    """
    # 1. Primary Alignment: The predicted vector should be close to the target
    # Since they are normalized, MSE is equivalent to optimizing Cosine Similarity
    mse_loss = F.mse_loss(zq, target_joint)

    # 2. Directional Penalty:
    # Ensure the 'move' we made from Q is in the right direction
    predicted_move = zq - original_q
    actual_move = target_joint - original_q

    # Cosine embedding loss forces the direction of the 'move' to align
    direction_loss = 1 - F.cosine_similarity(predicted_move, actual_move).mean()

    return mse_loss + 0.1 * direction_loss

def info_nce_loss(zq, target_joint, all_train_joint, temperature=0.07):
    """
    zq: [B, D] - Adapter output
    target_joint: [B, D] - The specific positive targets for this batch
    all_train_joint: [N, D] - The entire database of joint embeddings
    """
    # 1. Similarity to the correct target (Positive Logits)
    # Shape: [B]
    pos_sim = torch.sum(zq * target_joint, dim=-1) / temperature
    
    # 2. Similarity to EVERY joint embedding in the training set (All Logits)
    # Shape: [B, N]
    all_sims = (zq @ all_train_joint.T) / temperature
    
    # 3. LogSumExp over the N dimension (The denominator of Softmax)
    # This forces zq to be further from ALL negatives than it is to the positive
    loss = -pos_sim + torch.logsumexp(all_sims, dim=-1)
    
    return loss.mean()

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
    gold,
    label,
):

    print(f"\n===== {label} =====")

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

@torch.no_grad()
def evaluate_topk_alignment(test_q, test_s, test_joint, train_q, train_s, train_joint, gold, label):
    print(f"\n===== {label} (Top K Accuracy) =====")
    
    total = len(gold)
    # Ensure gold is on the correct device and reshaped for broadcasting: [N, 1]
    indices_gold = torch.topk(gold, k=5, dim=1).indices

    # 1. Get Top 5 for both variants
    # sim_qq: [N, N_train], sim_qj: [N, N_train]
    indices_qq = torch.topk(test_q @ train_q.T, k=5, dim=1).indices
    indices_qs = torch.topk(test_q @ train_s.T, k=5, dim=1).indices
    indices_qj = torch.topk(test_q @ train_joint.T, k=5, dim=1).indices
    
    acc_qq = (indices_qq[:, 0] == indices_gold[:, 0]).float().mean().item()
    acc_qs = (indices_qs[:, 0] == indices_gold[:, 0]).float().mean().item()
    acc_qj = (indices_qj[:, 0] == indices_gold[:, 0]).float().mean().item()
    print(f"Q->Q: {acc_qq}")
    print(f"Q->S: {acc_qs}")
    print(f"Q->J: {acc_qj}")

# ------------------------------------------------
# TRAIN ADAPTER
# ------------------------------------------------

def train_adapter(train_q, train_s, train_joint):

    adapter_q = SmallAdapter().to(DEVICE)
    adapter_s = None #SmallAdapter().to(DEVICE)
    adapter_joint = None#SmallAdapter().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(adapter_q.parameters()),
        #list(adapter_s.parameters()),
        #list(adapter_joint.parameters()),
        lr=LR,
    )

    N = train_q.size(0)
    #teacher_top1_idx = compute_teacher_targets(train_joint)
    teacher_top1_idx = torch.arange(N, device=DEVICE)

    print("\nTraining small adapter...")

    for epoch in range(EPOCHS):

        perm = torch.randperm(N)

        total_loss = 0

        for i in range(0, N, BATCH_SIZE):

            idx = perm[i:i+BATCH_SIZE]

            q = train_q[idx]
            joint = train_joint[idx]
            s = train_s[idx]

            zq = adapter_q(q)
            #zjoint = adapter_joint(joint)
            #zs = adapter_s(s)
            #zjoint = build_joint(zq, zs)

            #loss = contrastive_loss(zq, zjoint) + 0.1 * F.mse_loss(zjoint, joint)
            #loss = contrastive_loss(zq, joint)
            loss = kl_divergence_loss(
                zq @ train_joint.T, 
                joint @ train_joint.T, 
            )
            #loss = kl_divergence_loss(
            #    zq @ zq.T,
            #    joint @ joint.T
            #)
            #loss = 0.5 * contrast + 0.5 * div
            #loss = retrieval_distill_loss(
            #    zq, 
            #    train_joint, 
            #    teacher_top1_idx[idx]
            #)
            #loss = residual_difference_loss(zq, joint, q)
            #loss = info_nce_loss(zq, joint, train_joint)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} ",
                f"loss={total_loss/(N//BATCH_SIZE):.4f}"
            )

    return adapter_q, adapter_s, adapter_joint


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

    train_joint_text = [x[0] + x[1] for x in train] 
    test_joint_text = [x[0] + x[1] for x in test]

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

    #train_joint = build_joint(train_q, train_s)
    #test_joint  = build_joint(test_q, test_s)

    train_joint = normalize(embed_texts(
        train_joint_text,
        CACHE_DIR / "train_joint.npy"
    ))

    test_joint = normalize(embed_texts(
        test_joint_text,
        CACHE_DIR / "test_joint.npy"
    ))
    # GOLD STANDARD = joint semantic pair
    gold = nearest_neighbor(test_joint, train_joint)
    #gold = test_joint @ train_joint.T
    #gold = test_s @ train_s.T

    evaluate_alignment(
        test_q,
        test_s,
        test_joint,
        train_q,
        train_s,
        train_joint,
        gold,
        "BASELINE EMBEDDINGS"
    )

    # ------------------------------------------------
    # TRAIN ADAPTER
    # ------------------------------------------------

    #adapter_q, adapter_s = train_adapter(train_q, train_s)
    #adapter_q = train_adapter(train_q, train_s)
    adapter_q, adapter_s, adapter_joint = train_adapter(train_q, train_s, train_joint)

    # ------------------------------------------------
    # ADAPTED EMBEDDINGS
    # ------------------------------------------------

    with torch.no_grad():
        train_q_a = adapter_q(train_q)
        #train_joint_a = adapter_joint(train_joint)
        #train_s_a = adapter_s(train_s)

        test_q_a = adapter_q(test_q)
        #test_joint_a = adapter_joint(test_joint)
        #test_s_a = adapter_s(test_s)

    evaluate_alignment(
        test_q_a,
        test_s,
        test_joint,
        train_q_a,
        train_s,
        train_joint,
        gold,
        "AFTER SMALL ADAPTER"
    )
    torch.save(adapter_q.state_dict(), ADAPTER_PATH)
    print(f"Saved to {ADAPTER_PATH}")

# ------------------------------------------------

if __name__ == "__main__":
    main()
