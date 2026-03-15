import json
import random
import torch
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_NAME = "bert-base-uncased"

TRAIN_GEN = "data/WebQSP/generation/merged/WebQSP_train.json"
TRAIN_REL = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_train_cand_rels_sorted.json"
TRAIN_CROSS = "data/WebQSP/relation_retrieval/cross_encoder/rich_relation_3epochs_question_relation/WebQSP.train.tsv"
 

TEST_GEN = "data/WebQSP/generation/merged/WebQSP_test.json"
TEST_REL = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"
TEST_CROSS = "data/WebQSP/relation_retrieval/cross_encoder/rich_relation_3epochs_question_relation/WebQSP.test.tsv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 0
LR = 2e-5

NEGATIVE_RATIO = 8 
HARD_NEGATIVE_POOL = 100

MAX_LEN = 128

# ------------------------------------------------
# Gold relation extraction
# ------------------------------------------------

REL_PATTERN = re.compile(r"\[\s*([^\]]+)\]")

def extract_relations(expr):

    rels = []

    for match in REL_PATTERN.findall(expr):

        parts = [p.strip().replace(" ", "_") for p in match.split(",") if p.strip()]

        if len(parts) >= 3:
            rels.append(".".join(parts))

    return list(set(rels))

# ------------------------------------------------
# Dataset (Listwise)
# ------------------------------------------------

class ListwiseRelationDataset(Dataset):

    def __init__(self, gen_data, rel_data):

        self.data = []

        for qid in gen_data:

            question = gen_data[qid]["question"]
            expr = gen_data[qid]["normed_sexpr"]

            gold = set(extract_relations(expr))
            candidates = rel_data[qid]

            positives = [r for r, s in candidates if r in gold]

            if len(positives) == 0:
                continue

            pos = positives[0]

            hard_negs = [r for r, s in candidates[:HARD_NEGATIVE_POOL] if r not in gold]

            if len(hard_negs) < NEGATIVE_RATIO:
                hard_negs = [r for r, s in candidates if r not in gold]

            negatives = random.sample(
                hard_negs,
                min(len(hard_negs), NEGATIVE_RATIO)
            )

            relations = [pos] + negatives

            self.data.append((question, relations))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------------------------------
# Model
# ------------------------------------------------

class CrossEncoder(torch.nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        self.classifier = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
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
    group_sizes = []

    for q, rels in batch:

        group_sizes.append(len(rels))

        for r in rels:
            questions.append(q)
            relations.append(r)

    enc = tokenizer(
        questions,
        relations,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    return enc, group_sizes

# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data(gen_file, rel_file, cross_file):

    with open(gen_file) as f:
        gen_raw = json.load(f)

    gen_data = {x["ID"]: x for x in gen_raw}

    with open(rel_file) as f:
        rel_data = json.load(f)

    cross_data = {}
    if cross_file:
        cross_data_raw = pd.read_csv(cross_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
        for (idx, question, relation, label) in cross_data_raw.to_numpy():
            if question not in cross_data:
                cross_data[question] = []
            if label == 0:
                continue
            cross_data[question].append(relation)

    return gen_data, rel_data, cross_data

# ------------------------------------------------
# Training
# ------------------------------------------------

def train_model():

    gen_data, rel_data, cross_data = load_data(TEST_GEN, TEST_REL, TEST_CROSS)

    dataset = ListwiseRelationDataset(gen_data, rel_data)

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

        for enc, group_sizes in tqdm(loader):

            enc = {k:v.to(DEVICE) for k,v in enc.items()}

            scores = model(**enc)

            start = 0
            losses = []

            for size in group_sizes:

                group_scores = scores[start:start+size]

                target = torch.zeros(
                    1,
                    dtype=torch.long,
                    device=DEVICE
                )

                group_scores = group_scores.unsqueeze(0)

                loss = F.cross_entropy(group_scores, target)

                losses.append(loss)

                start += size

            loss = torch.stack(losses).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print("Epoch", epoch, "loss:", total_loss/len(loader))

    return model, tokenizer

# ------------------------------------------------
# Prediction
# ------------------------------------------------

def predict_relations(question, candidates, model, tokenizer):

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

    probs = torch.sigmoid(scores).cpu().numpy()

    selected = [
        r for r,p in zip(candidates, probs)
        if p >= THRESHOLD
    ]

    return selected

# ------------------------------------------------
# Metrics
# ------------------------------------------------

def compute_counts(predicted, gold):

    predicted = set(predicted)
    gold = set(gold)

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp>0 else 0
    recall = tp/(tp+fn) if tp+fn>0 else 0

    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0

    return precision, recall, f1

# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(model, tokenizer):

    gen_data, rel_data, cross_data = load_data(TEST_GEN, TEST_REL, TEST_CROSS)

    model.eval()

    for threshold in [0.15]:
        
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for qid in tqdm(gen_data):

            question = gen_data[qid]["question"]

            expr = gen_data[qid]["normed_sexpr"]

            gold_extracted = extract_relations(expr)
            gold_labeled = cross_data[question]

            candidates = rel_data[qid]

        #predicted = predict_relations(
        #    question,
        #    candidates,
        #    model,
        #    tokenizer
        #)
            scores = torch.softmax(torch.tensor([score for _, score in candidates]), dim=-1)
            predicted = [ rel for i, (rel, _) in enumerate(candidates) if scores[i] >= threshold ]
            #predicted = [ rel for rel, _ in candidates[:len(gold_extracted)] ]

            #if set(predicted) != set(gold_extracted):
                #print(f"Candidates: {candidates[:10]}")
                #print(f"Predicted: {predicted}")
                #print(f"Groundtruth: {gold_extracted}")
                #predicted = predicted[1:len(gold_extracted)+1]

            tp, fp, fn = compute_counts(predicted, gold_extracted)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        print(f"\nEvaluation Results for threshold={threshold}")
    
        precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

        print({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    model, tokenizer = train_model()

    evaluate(model, tokenizer)

if __name__ == "__main__":
    main()
