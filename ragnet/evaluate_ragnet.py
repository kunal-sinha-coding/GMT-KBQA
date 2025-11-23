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
from relation_retrieval.bi_encoder.run_bi_encoder import full_system_prompt

BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
load_dotenv()

LLM_NAME = "meta-llama/Llama-2-7b-chat-hf"
GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"
RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"

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


def evaluate_single(llm_model, llm_tokenizer, example, top_k=2):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    question, relations, answer = example["question"], example["relations"], example["answer"]
    #prompt = f"{full_system_prompt}\nQuestion: {question}\nRelevant relations: {relations[:top_k]}\nLogical form: "
    #inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    #outputs = llm_model.generate(
    #    **inputs
    #)
    #normed_sexpr = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # denormalize_s_expr_new(
    #     normed_expr, 
    #     entity_label_map,
    #     type_label_map,
    #     rel_label_map,
    #     train_entity_map,
    #     surface_index
    # )
    #TODO: Convert normed_sexpr into SPARQL query
    #TODO: Make this asynchronous with semaphore
    sparql_query = (
        """
        SELECT ?river ?riverLabel ?length WHERE {
        ?river wdt:P31 wd:Q4022;
            wdt:P30 wd:Q15;
            wdt:P2043 ?length.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        ORDER BY (?length)
        LIMIT 1
        """
    )
    import pdb; pdb.set_trace()
    groundtruth = query_database_with_sparql()
    predictions = query_database_with_sparql(sparql_query)
    tp, fp, fn = get_retrieval_counts(predictions, groundtruth)
    return tp, fp, fn

def get_retrieval_counts(predictions, groundtruth):
    predictions, groundtruth = set(predictions), set(groundtruth)
    tp = sum(predictions.intersection(groundtruth))
    fp = len(predictions) - tp
    fn = len(groundtruth) - tp
    return tp, fp, fn


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
            predictions.append(res[label_key]["value"])
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


def main():
    llm_model, llm_tokenizer = None, None #load_llm_and_tokenizer()
    data = load_data()
    for current_id, example in data.items():
        import pdb; pdb.set_trace()
        evaluate_single(llm_model, llm_tokenizer, example)
    

if __name__ == "__main__":
    main()    
