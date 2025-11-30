import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import copy
import argparse
from dotenv import load_dotenv
from itertools import chain
import json
import wandb
import time
import httpx
import asyncio
from executor.sparql_executor import execute_query_with_odbc
from relation_retrieval.bi_encoder.run_bi_encoder import full_system_prompt
from components.utils import load_json
from entity_retrieval import surface_index_memory
from eval_topk_prediction_final import denormalize_s_expr_new
from executor.logic_form_util import lisp_to_sparql

BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
load_dotenv()

LLM_NAME = "meta-llama/Llama-2-7b-chat-hf"
GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"
RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"

TRAIN_ENTITY_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_entity_label_map.json"
CANDIDATE_ENTITY_MAP_NAME = "data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_test_disamb_entities.json"
TRAIN_RELATION_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_relation_label_map.json"
TRAIN_TYPE_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json"

def load_data():
    data = {}
    with open(RELATIONS_DATA_NAME, "r") as f:
        relations_data = json.load(f)
    with open(GENERATION_DATA_NAME) as f:
        generation_data_raw = json.loads(f.read())
        generation_data = {
            gen_data['ID']: gen_data
            for gen_data in generation_data_raw
        }
    for current_id in generation_data:
        data[current_id] = {
            **generation_data[current_id],
            "relations": relations_data[current_id]
        }
    return data


def evaluate_single(llm_model, llm_tokenizer, example, database_info, top_k=2):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    question, relations, answer = example["question"], example["relations"], example["answer"]
    #prompt = f"{full_system_prompt}\nQuestion: {question}\nRelevant relations: {relations[:top_k]}\nLogical form: "
    #inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    #outputs = llm_model.generate(
    #    **inputs
    #)
    #normed_sexpr = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    normed_expr = "( JOIN [ organization , founder , person ] [ Microsoft ] )"
    #TODO: Make this asynchronous with semaphore
    #sparql_query = (
    #    """
    #    SELECT ?name WHERE {
    #	    fb:m.044tg ?p ?o .
    #        ?o fb:type.object.name ?name .
    #    }
    #    LIMIT 10 
    #    """
    #)
    sparql_query = convert_normed_expr_to_sparql(normed_expr, example["ID"], database_info)
    results = execute_query_with_odbc(sparql_query)
    predictions = [ res.split("/")[-1] for res in results ]
    tp, fp, fn = get_retrieval_counts(predictions, answer)
    print(f"Retrieval counts: {tp}, {fp}, {fn}")
    return tp, fp, fn

def convert_normed_expr_to_sparql(normed_expr, question_id, database_info):
    entity_label_map = {}
    if question_id in database_info["candidate_entity_map"]:
        entity_label_map = {
            item["label"].lower(): item["id"] 
            for item in database_info["candidate_entity_map"][question_id]
        }
    denorm_expr = denormalize_s_expr_new(
        normed_expr,
        entity_label_map,
        database_info["type_label_map"],
        database_info["rel_label_map"],
        database_info["train_entity_map"],
        database_info["surface_index"]
    )
    query_expr = denorm_expr.replace("( ", "(").replace(" )", ")")
    return lisp_to_sparql(query_expr)


def get_retrieval_counts(predictions, groundtruth):
    predictions, groundtruth = set(predictions), set(groundtruth)
    tp = len(predictions.intersection(groundtruth))
    fp = len(predictions) - tp
    fn = len(groundtruth) - tp
    return tp, fp, fn

def calculate_retrieval_metrics(tp, fp, fn):
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


def query_database_with_sparql(sparql_query):
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    with httpx.Client() as client:
        response = client.get(url, params={"query": sparql_query}, headers=headers)
        data = response.json()
        results = [item for item in data["results"]["bindings"]]
        predictions = []
        for res in results:
            label_key = [k for k in res.keys() if "Label" in k][0]
            predictions.append(res[label_key]["value"].split("/")[-1])
        return predictions

    
def load_llm_and_tokenizer():
    # New code for loading in LLM
    llm_token = os.getenv("HF_AUTH_TOKEN")
    print(f"Loading in LLM and tokenizer: {LLM_NAME}...")
    before_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"Before loading in LLM: {before_mem:.2f} GB of CUDA memory used")
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.bfloat16,
        use_auth_token=llm_token,
        device_map="auto"
    )
    after_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"After loading in LLM: {after_mem:.2f} GB of CUDA memory used")
    print(f"Total LLM CUDA memory usage: {(after_mem - before_mem):.2f} GB")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME, use_fast=False,
        use_auth_token=llm_token,
        device_map="auto"
    )
    llm_tokenizer.add_special_tokens({"pad_token": LLM_PAD_TOKEN})
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    print("LLM tokenizer successfully loaded")
    return llm_model, llm_tokenizer


def load_database_info():

    train_entity_map = load_json(TRAIN_ENTITY_MAP_NAME)
    candidate_entity_map = load_json(CANDIDATE_ENTITY_MAP_NAME)
    train_relation_map = load_json(TRAIN_RELATION_MAP_NAME)
    train_type_map = load_json(TRAIN_TYPE_MAP_NAME)

    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention",
        "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention"
    )
    return {
        "train_entity_map": {l.lower(): e for e, l in train_entity_map.items()},
        "type_label_map": {l.lower(): t for t, l in train_type_map.items()},
        "rel_label_map": {l.lower(): r for r, l in train_relation_map.items()},
        "candidate_entity_map": candidate_entity_map,
        "surface_index": surface_index
    }
 

def main():
    llm_model, llm_tokenizer = None, None #load_llm_and_tokenizer()
    data = load_data()
    database_info = load_database_info()
    total_tp, total_fp, total_fn = 0, 0, 0
    for current_id, example in data.items():
        tp, fp, fn = evaluate_single(llm_model, llm_tokenizer, example, database_info)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    print(calculate_retrieval_metrics(total_tp, total_fp, total_fn))


if __name__ == "__main__":
    main()    
