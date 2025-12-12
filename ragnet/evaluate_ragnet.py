import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import copy
import argparse
from dotenv import load_dotenv
from itertools import chain
import json
import wandb
import time
import asyncio
from tqdm.asyncio import tqdm_asyncio
from executor.sparql_executor import execute_query_with_odbc
from relation_retrieval.bi_encoder.consts import full_system_prompt
from components.utils import load_json
from entity_retrieval import surface_index_memory
from eval_topk_prediction_final import denormalize_s_expr_new
from executor.logic_form_util import lisp_to_sparql
from pathlib import Path
import time
import re

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

OUTPUT_FILE = Path("ragnet/outputs.txt")

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

async def evaluate_all(data, database_info, llm_model, llm_tokenizer, device, batch_size=1):
    all_examples = list(data.values())
    stopping_criteria = StoppingCriteriaList([StopOnMultipleWords(["question", "q:"], llm_tokenizer)])
    results = []
    for i in tqdm(range(len(all_examples)), desc="Evaluating"):
        start, end = i * batch_size, (i + 1) * batch_size
        examples_batch = all_examples[start:end]
        try:
            tp, fp, fn = await evaluate_single(
                llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info
            )
            print(tp, fp, fn)
            results.append((tp, fp, fn))
        except Exception as error:
            print(f"Error: {error}")
    tp, fp, fn = map(list, zip(*results))
    return sum(tp), sum(fp), sum(fn)

class StopOnMultipleWords(StoppingCriteria):
    def __init__(self, stop_words, llm_tokenizer):
        self.stop_words = stop_words
        self.llm_tokenizer = llm_tokenizer
        self.last_n_tokens = 3

    def __call__(self, input_ids, scores, **kwargs):
        text = self.llm_tokenizer.decode(input_ids[0, -self.last_n_tokens:], skip_special_tokens=True).lower()
        return any(word.lower() in text for word in self.stop_words)

async def evaluate_single(llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info, top_k=5):

    # Generate logical form with LLM
    prompts = []
    for example in examples_batch:
        question, question_id, relations, answer = example["question"], example["ID"], example["relations"], example["answer"]
        prompts.append(f"{full_system_prompt}\nQuestion: {question}\nRelations: {relations[:top_k]}\nLogical form: ")
    all_normed_expr = get_normed_expr(llm_model, llm_tokenizer, device, stopping_criteria, prompts)
    
    # Convert logical forms into SPARQL and query the database
    results = await asyncio.gather(*[
        query_database(all_normed_expr[i], example, database_info)
        for i, example in enumerate(examples_batch)
    ], return_exceptions=True)
    tp, fp, fn = map(list, zip(*results))
    return sum(tp), sum(fp), sum(fn)


def get_normed_expr(llm_model, llm_tokenizer, device, stopping_criteria, prompts):
    inputs = llm_tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    outputs = llm_model.generate(
        **inputs,
        stopping_criteria=stopping_criteria,
        max_new_tokens=100
    )
    decoded_outputs = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_normed_expr = []
    for decoded in decoded_outputs:
        decoded = decoded[ decoded.index("Logical form:") :].strip()
        s = decoded[ decoded.index("(") : decoded.rfind(")") + 1]
        s = re.sub(r'([\[\]\(\),])', r' \1 ', s)
        normed_expr = re.sub(r'\s+', ' ', s).strip()
        all_normed_expr.append(normed_expr)
    return all_normed_expr


async def query_database(normed_expr, example, database_info):
    
    # Query the database
    question, question_id, relations, answer = example["question"], example["ID"], example["relations"], example["answer"] 
    sparql_query = convert_normed_expr_to_sparql(normed_expr, question_id, database_info)
    results = execute_query_with_odbc(sparql_query)
    predictions = [ res.split("/")[-1] for res in results ]
    import pdb; pdb.set_trace()

    # Compute evaluation metrics and save
    tp, fp, fn = get_retrieval_counts(predictions, answer)
    with OUTPUT_FILE.open("a") as output_file:
        output = f"Question ID: {question_id}"
        output += f"\nTP: {tp}, FP: {fp}, FN: {fn}"
        output += f"\nQuestion: {question}\nAnswer: {answer}"
        output += f"\nRelations: {relations}\nPredictions: {predictions}"
        output_file.write(f"Retrieval counts: {tp}, {fp}, {fn}")
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


#def query_database_with_sparql(sparql_query):
#    url = "https://query.wikidata.org/sparql"
#    headers = {"Accept": "application/sparql-results+json"}
#    with httpx.Client() as client:
#        response = client.get(url, params={"query": sparql_query}, headers=headers)
#        data = response.json()
#        results = [item for item in data["results"]["bindings"]]
#        predictions = []
#        for res in results:
#            label_key = [k for k in res.keys() if "Label" in k][0]
#            predictions.append(res[label_key]["value"].split("/")[-1])
#        return predictions

    
def load_llm_and_tokenizer():
    # New code for loading in LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    llm_token = os.getenv("HF_AUTH_TOKEN")
    print(f"Loading in LLM and tokenizer: {LLM_NAME}...")
    before_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"Before loading in LLM: {before_mem:.2f} GB of CUDA memory used")
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16,
        use_auth_token=llm_token,
    ).to(device)
    after_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"After loading in LLM: {after_mem:.2f} GB of CUDA memory used")
    print(f"Total LLM CUDA memory usage: {(after_mem - before_mem):.2f} GB")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME, use_fast=False,
        use_auth_token=llm_token,
    )
    llm_tokenizer.add_special_tokens({"pad_token": LLM_PAD_TOKEN})
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    print("LLM tokenizer successfully loaded")
    return llm_model, llm_tokenizer, device


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
 

async def main():
    start_time = time.time()
    llm_model, llm_tokenizer, device = load_llm_and_tokenizer()
    data = load_data()
    database_info = load_database_info()
    print(f"Startup time: {time.time() - start_time}")
    with torch.no_grad():
        total_tp, total_fp, total_fn = await evaluate_all(data, database_info, llm_model, llm_tokenizer, device)
    calculate_retrieval_metrics(total_tp, total_fp, total_fn)


if __name__ == "__main__":
    asyncio.run(main())   
