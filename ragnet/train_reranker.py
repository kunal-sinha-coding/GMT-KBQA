import json
import random
import torch
import numpy as np
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_NAME = "sentence-transformers/ms-marco-MiniLM-L-6-v2"

TRAIN_GEN = "data/WebQSP/generation/merged/WebQSP_train.json"
TRAIN_REL = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_train_cand_rels_sorted.json"

TEST_GEN = "data/WebQSP/generation/merged/WebQSP_test.json"
TEST_REL = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
NEGATIVE_RATIO = 3
MAX_LEN = 128

K_VALUES = [1,3,5,10,20]


# ------------------------------------------------
# Gold relation extraction
# ------------------------------------------------

REL_PATTERN = re.compile(r"\[\s*([^\]]+)\]")

def extract_relations(expr):

    rels = []

    for match in REL_PATTERN.findall(expr):

        if "," in match:
            parts = [p.strip() for p in match.split(",")]
            rels.append(".".join(parts))

    return list(set(rels))


# ------------------------------------------------
# Dataset
# ------------------------------------------------

class RelationDataset(Dataset):

    def __init__(self, gen_data, rel_data):

        self.samples = []

        for qid in gen_data:

            question = gen_data[qid]["question"]
            expr = gen_data[qid]["normed_sexpr"]

            gold = set(extract_relations(expr))
            candidates = rel_data[qid]

            positives = [r for r in candidates if r in gold]
            negatives = [r for r in candidates if r not in gold]

            if len(positives) == 0:
                continue

            for pos in positives:

                self.samples.append((question, pos, 1))

                for _ in range(NEGATIVE_RATIO):

                    if len(negatives) == 0:
                        break

                    neg = random.choice(negatives)

                    self.samples.append((question, neg, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------------------------
# Model
# ------------------------------------------------

class CrossEncoder(torch.nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.classifier = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0]

        score = self.classifier(cls)

        return score.squeeze(-1)


# ------------------------------------------------
# Collate
# ------------------------------------------------

def collate(batch, tokenizer):

    questions = []
    relations = []
    labels = []

    for q,r,y in batch:
        questions.append(q)
        relations.append(r)
        labels.append(y)

    enc = tokenizer(
        questions,
        relations,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    labels = torch.tensor(labels, dtype=torch.float)

    return enc, labels


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data(gen_file, rel_file):

    with open(gen_file) as f:
        gen_raw = json.load(f)

    gen_data = {x["ID"]: x for x in gen_raw}

    with open(rel_file) as f:
        rel_data = json.load(f)

    return gen_data, rel_data


# ------------------------------------------------
# Training
# ------------------------------------------------

def train_model():

    gen_data, rel_data = load_data(TRAIN_GEN, TRAIN_REL)

    dataset = RelationDataset(gen_data, rel_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate(x, tokenizer)
    )

    model = CrossEncoder(MODEL_NAME).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for enc, labels in tqdm(loader):

            enc = {k:v.to(DEVICE) for k,v in enc.items()}
            labels = labels.to(DEVICE)

            scores = model(**enc)

            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print("Epoch", epoch, "loss:", total_loss/len(loader))

    return model, tokenizer


# ------------------------------------------------
# Reranking
# ------------------------------------------------

def rerank(question, candidates, model, tokenizer):

    enc = tokenizer(
        [question]*len(candidates),
        candidates,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    enc = {k:v.to(DEVICE) for k,v in enc.items()}

    with torch.no_grad():

        scores = model(**enc)

    scores = scores.cpu().numpy()

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in ranked]


# ------------------------------------------------
# Metrics
# ------------------------------------------------

def compute_metrics(retrieved, gold):

    retrieved = set(retrieved)
    gold = set(gold)

    tp = len(retrieved & gold)
    fp = len(retrieved - gold)
    fn = len(gold - retrieved)

    precision = tp/(tp+fp) if tp+fp>0 else 0
    recall = tp/(tp+fn) if tp+fn>0 else 0

    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0

    hits = int(tp>0)

    return precision, recall, f1, hits


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(model, tokenizer):

    gen_data, rel_data = load_data(TEST_GEN, TEST_REL)

    results = {k:[] for k in K_VALUES}

    model.eval()

    for qid in tqdm(gen_data):

        question = gen_data[qid]["question"]

        expr = gen_data[qid]["normed_sexpr"]

        gold = extract_relations(expr)

        candidates = rel_data[qid]

        ranked = rerank(question, candidates, model, tokenizer)

        for k in K_VALUES:

            topk = ranked[:k]

            results[k].append(
                compute_metrics(topk, gold)
            )

    for k in results:

        arr = np.array(results[k])

        precision = arr[:,0].mean()
        recall = arr[:,1].mean()
        f1 = arr[:,2].mean()
        hits = arr[:,3].mean()

        print("\nK =", k)

        print({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hits": hits
        })


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    model, tokenizer = train_model()

    evaluate(model, tokenizer)


if __name__ == "__main__":
    main()
