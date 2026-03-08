import json
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path

TEST_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"
TEST_RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"
TEST_ENTITY_DATA_NAME = "data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test_merged_cand_entities_elq_facc1.json"


K_VALUES = [1, 3, 5, 10, 20, 50, -1]


def load_data():
    with open(TEST_GENERATION_DATA_NAME) as f:
        generation_data_raw = json.load(f)

    generation_data = {
        item["ID"]: item for item in generation_data_raw
    }

    with open(TEST_RELATIONS_DATA_NAME) as f:
        relations_data = json.load(f)

    with open(TEST_ENTITY_DATA_NAME) as f:
        entity_data = json.load(f)

    data = {}

    for qid in generation_data:
        data[qid] = {
            "question": generation_data[qid]["question"],
            "normed_sexpr": generation_data[qid]["normed_sexpr"],
            "candidate_relations": relations_data[qid],
            "candidate_entities": [e["label"] for e in entity_data[qid]]
        }

    return data


# ---------------------------------------------------------
# Logical form parsing
# ---------------------------------------------------------

RELATION_PATTERN = re.compile(r"\[\s*([^\]]+?)\s*\]")
ENTITY_PATTERN = re.compile(r"\[\s*([^\]]+?)\s*\]")


def extract_relations(expr):
    """
    Extract relations from normalized S-expression.
    Converts: [ location , containedby ]
    into: location.containedby
    """
    relations = []

    for match in RELATION_PATTERN.findall(expr):
        parts = [p.strip() for p in match.split(",")]
        rel = ".".join(parts)
        relations.append(rel)

    return relations


def extract_entities(expr):
    """
    Extract entities appearing inside [] that are not relations.
    In WebQSP logical forms, entities are also bracketed.
    """
    entities = []

    for match in ENTITY_PATTERN.findall(expr):
        if "," not in match:
            entities.append(match.strip())

    return entities


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_metrics(retrieved, gold):

    retrieved = set(retrieved)
    gold = set(gold)

    tp = len(retrieved & gold)
    fp = len(retrieved - gold)
    fn = len(gold - retrieved)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)

    hits = int(tp > 0)

    return precision, recall, f1, hits

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_retrieval(data):

    entity_results = {k: [] for k in K_VALUES}
    relation_results = {k: [] for k in K_VALUES}

    for example in tqdm(data.values()):

        expr = example["normed_sexpr"]

        gold_entities = extract_entities(expr)
        gold_relations = extract_relations(expr)

        cand_entities = example["candidate_entities"]
        cand_relations = example["candidate_relations"]

        for k in K_VALUES:

            topk_entities = cand_entities[:k]
            topk_relations = cand_relations[:k]

            entity_results[k].append(
                compute_metrics(topk_entities, gold_entities)
            )

            relation_results[k].append(
                compute_metrics(topk_relations, gold_relations)
            )

    return entity_results, relation_results


def aggregate(results):

    metrics = {}

    for k, values in results.items():

        arr = np.array(values)

        precision = arr[:,0].mean()
        recall = arr[:,1].mean()
        f1 = arr[:,2].mean()
        hits = arr[:,3].mean()

        metrics[k] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hits": hits
        }

    return metrics


def main():

    data = load_data()

    entity_results, relation_results = evaluate_retrieval(data)

    entity_metrics = aggregate(entity_results)
    relation_metrics = aggregate(relation_results)

    print("\nEntity Retrieval Metrics")
    print(json.dumps(entity_metrics, indent=2))
    import pdb; pdb.set_trace()

    print("\nRelation Retrieval Metrics")
    print(json.dumps(relation_metrics, indent=2))


if __name__ == "__main__":
    main()
