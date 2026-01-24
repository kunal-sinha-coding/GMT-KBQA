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
from ragnet.prompts import system_prompt_lambda_dcs_type_v2
from components.utils import load_json
from entity_retrieval import surface_index_memory
from eval_topk_prediction_final import denormalize_s_expr_new
from executor.logic_form_util import lisp_to_sparql
from pathlib import Path
import time
import re
from openai import AsyncOpenAI
import asyncio

BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
load_dotenv()


GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"
RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"
ENTITY_DATA_NAME = "data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test_merged_cand_entities_elq_facc1.json"

TRAIN_ENTITY_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_entity_label_map.json"
CANDIDATE_ENTITY_MAP_NAME = "data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_test_disamb_entities.json"
TRAIN_RELATION_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_relation_label_map.json"
TRAIN_TYPE_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json"

OUTPUT_FILE = Path("ragnet/outputs.txt")
RESULTS_FILE = Path("ragnet/results.jsonl")

LLM_MODEL_NAME = "gpt-5-nano" #"gpt-5.2-2025-12-11" 
#"meta-llama/Llama-3.1-8B" #"meta-llama/Llama-2-7b-chat-hf"
LLM_MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-5.2-2025-12-11": {
        "input": 1.75 / 1_000_000,
        "output": 14.00 / 1_000_000,
    },
    "gpt-5-nano": {
        "input": 0.05 / 1_000_000,
        "output": 0.40 / 1_000_000
    }
}
openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

def load_data():
    data = {}
    with open(RELATIONS_DATA_NAME, "r") as f:
        relations_data = json.load(f)
    with open(ENTITY_DATA_NAME, "r") as f:
        entity_data = json.load(f)
    with open(GENERATION_DATA_NAME) as f:
        generation_data_raw = json.loads(f.read())
        generation_data = {
            gen_data['ID']: gen_data
            for gen_data in generation_data_raw
        }
    for current_id in generation_data:
        data[current_id] = {
            **generation_data[current_id],
            "relations": relations_data[current_id],
            "entities": [ entity["label"] for entity in entity_data[current_id] ]
        }
    return data

async def evaluate_all(data, database_info, llm_model, llm_tokenizer, device, batch_size=1):
    all_examples = list(data.values())
    stopping_criteria = StoppingCriteriaList([StopOnMultipleWords(["question", "q:"], llm_tokenizer)])
    all_results = []
    for i in tqdm(range(len(all_examples) // batch_size), desc="Evaluating"):
        start, end = i * batch_size, (i + 1) * batch_size
        examples_batch = all_examples[start:end]
        result = await evaluate_single(
            llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info
        )
        all_results.append(result)
    tp, fp, fn, hits1, hits, count = map(list, zip(*all_results))
    return sum(tp), sum(fp), sum(fn), sum(hits1), sum(hits), sum(count)

class StopOnMultipleWords(StoppingCriteria):
    def __init__(self, stop_words, llm_tokenizer):
        self.stop_words = stop_words
        self.llm_tokenizer = llm_tokenizer
        self.last_n_tokens = 3

    def __call__(self, input_ids, scores, **kwargs):
        text = self.llm_tokenizer.decode(input_ids[0, -self.last_n_tokens:], skip_special_tokens=True).lower()
        return any(word.lower() in text for word in self.stop_words)

async def evaluate_single(llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info, top_k=5):

    # Get predictions
    prompts = []
    for example in examples_batch:
        question, question_id, entities, relations, answer = example["question"], example["ID"], example["entities"], example["relations"], example["answer"]
        prompts.append(f"\nQuestion: {question}\nEntities: {entities[:top_k]}\nRelations: {relations[:top_k]}\nLogical form: ")
        #prompts.append(f"Question:{question}\nLogical form: ")
    all_normed_expr, all_sparql_queries, all_predictions = await get_predictions_gpt(prompts, question_id, database_info)
    #get_predictions(llm_model, llm_tokenizer, device, stopping_criteria, prompts, question_id, database_info)

    # Compute evaluation metrics and save
    total_tp, total_fp, total_fn, total_hits1, total_hits, total_count = 0, 0, 0, 0, 0, 0
    for i, example in enumerate(examples_batch):
        question, question_id, relations, answer = example["question"], example["ID"], example["relations"], example["answer"]
        gt_normed_expr, gt_sparql_query = example["normed_sexpr"], example["sparql"]
        tp, fp, fn, hits1, hits = get_retrieval_counts(all_predictions[i], answer)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_hits1 += int(hits1)
        total_hits += int(total_hits)
        total_count += 1
        with OUTPUT_FILE.open("a") as output_file:
            output = f"Question ID: {question_id}"
            output += f"\nTP: {total_tp}, FP: {total_fp}, FN: {total_fn}"
            output += f"\nHits@1: {hits1}, Hits: {hits}"
            output += f"\nQuestion: {question}\nEntities: {entities[:top_k]}\nRelations: {relations[:top_k]}"
            output += f"\nPredicted normed expr: {all_normed_expr[i]}"
            output += f"\nPredicted query: {all_sparql_queries[i]}"
            output += f"\nPredictions: {all_predictions[i]}"
            output += f"\nGroundtruth normed expr: {gt_normed_expr}"
            output += f"\nGroundtruth query: {gt_sparql_query}"
            output += f"\nAnswer: {answer}\n\n"
            output_file.write(output)
    return total_tp, total_fp, total_fn, total_hits1, total_hits, total_count


async def get_normed_expr_gpt(prompt: str):
    content, usage = None, None
    try:
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt_lambda_dcs_type_v2},
                {"role": "user", "content": prompt},
            ]
        )

        usage = response.usage
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed GPT generation: {e}")
    return content, usage

def post_process_normed_expr_gpt(expr):
    expr = expr.replace('_', ' ') # remove underscores
    expr = expr.replace('\n', ' ').replace('\t', ' ')   # remove newlines and tabs
    expr = re.sub(r'\s+', ' ', expr).strip() # collapse multiple spaces
    return expr


async def get_predictions_gpt(prompts, question_id, database_info):
    all_normed_expr = [None for _ in prompts]
    all_sparql_queries = [None for _ in prompts]
    all_predictions = [[] for _ in prompts]

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = [get_normed_expr_gpt(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    for i, (decoded, usage) in enumerate(results):
        try:
            normed_expr = post_process_normed_expr_gpt(decoded)
        except Exception as e:
            print(f"Failed to post-process {decoded}: {e}")
            continue
        all_normed_expr[i] = normed_expr

        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens

        if not normed_expr:
            print(f"No normed expression from decoded: {decoded}")
            continue

        try:
            sparql_query = convert_normed_expr_to_sparql(
                normed_expr, question_id, database_info
            )
        except Exception as e:
            print(f"Converting {decoded} to SPARQL query failed: {e}")
            continue
        if not sparql_query:
            print(f"SPARQL query for {decoded} is empty")
            continue
        all_sparql_queries[i] = sparql_query
        
        try:
            results = execute_query_with_odbc(sparql_query)
        except Exception as e:
            print(f"Executing query {sparql_query} failed: {e}")
            continue
        try:
            predictions = [
                res.split("/")[-1] if res else None
                for res in results
            ]
        except Exception as e:
            print(f"Post processing results {results} failed: {e}")
            continue
        all_predictions[i] = predictions

    pricing = LLM_MODEL_PRICING[LLM_MODEL_NAME]
    total_cost = (
        total_input_tokens * pricing["input"]
        + total_output_tokens * pricing["output"]
    )
    # print(total_cost)

    return all_normed_expr, all_sparql_queries, all_predictions

    

def get_predictions(llm_model, llm_tokenizer, device, stopping_criteria, prompts, question_id, database_info):
    inputs = llm_tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    max_retries = 3
    retry_count = 0
    all_normed_expr = [ None for prompt in prompts ]
    all_sparql_queries = [ None for prompt in prompts ]
    all_predictions = [ [] for prompt in prompts ]
    while ((
        not all(all_normed_expr)
        or not all(all_sparql_queries)
        or not all(all_predictions)
    ) and retry_count < max_retries):
        if retry_count > 0:
            print(f"Retry count: {retry_count}")
        retry_count += 1
        try:
            outputs = llm_model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                max_new_tokens=1000
            )
        except Exception as e:
            print(f"ERROR: Failed in generation: {e}")
            continue
        try:
            decoded_outputs = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            print(f"ERROR: Failed in decoding: {e}")
            continue
        for i, decoded in enumerate(decoded_outputs):
            try:
                normed_expr = post_process_normed_expr(decoded)
            except Exception as e:
                print(f"ERROR: Failed post-process normed_expr: {e}")
                break
            if not normed_expr:
                print(f"ERROR: Failed post process normed_expr. Original: {decoded}")
                break
            try:
                sparql_query = convert_normed_expr_to_sparql(normed_expr_corrected, question_id, database_info)
                all_sparql_queries[i] = sparql_query
            except Exception as e:
                print(f"ERROR: Failed convert to sparql query: {e}")
                break
            try:
                results = execute_query_with_odbc(sparql_query)
                predictions = [ 
                    res.split("/")[-1] if res else None
                    for res in results 
                ]
                all_predictions[i] = predictions
            except Exception as e:
                print(f"ERROR: Failed execute sparql query: {e}")
                break
            if not predictions or not all(predictions):
                print(f"ERROR: Empty list of predictions. Predictions: {predictions}")
                break
    return all_normed_expr, all_sparql_queries, all_predictions


def post_process_normed_expr(decoded: str):
    
    # Remove fluff
    decoded = decoded[ decoded.index("Logical form:") :].strip()
    s = decoded[ decoded.index("(") : decoded.rfind(")") + 1]
    s = re.sub(r'([\[\]\(\),])', r' \1 ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Get all top level expressions
    exprs = []
    depth = 0
    start = None
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start is not None:
                exprs.append(s[start : i + 1].strip())
                start = None
    
    # Merge top levels with OR
    cleaned = []
    for e in exprs:
        e = e.strip()
        e = e.replace("_", " ") # Replace _ with spaces
        if e and e not in cleaned:
            cleaned.append(e)
    if len(cleaned) == 0:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    inner = "\n    ".join(cleaned)
    expr = f"( OR\n    {inner}\n)"
    
    # Remove exceess whitespace
    expr = expr.replace('\n', ' ').replace('\t', ' ')   # remove newlines and tabs
    expr = re.sub(r'\s+', ' ', expr).strip() # collapse multiple spaces
    return expr


def convert_normed_expr_to_sparql(normed_expr, question_id, database_info):
    entity_label_map = {}
    if question_id in database_info["candidate_entity_map"]:
        entity_label_map = {
            item["label"].lower(): item["id"] 
            for item in database_info["candidate_entity_map"][question_id]
        }
    try:
        denorm_expr = denormalize_s_expr_new(
            normed_expr,
            entity_label_map,
            database_info["type_label_map"],
            database_info["rel_label_map"],
            database_info["train_entity_map"],
            database_info["surface_index"]
        )
    except Exception as e:
        print(f"ERROR: Failed in denormalizing expression: {e}")
        return None
    query_expr = denorm_expr.replace("( ", "(").replace(" )", ")")
    try:
        sparql_query = lisp_to_sparql(query_expr)
    except Exception as e:
        print(f"ERROR: Failed in converting LISP to SPARQL: {e}")
        return None
    return sparql_query


def get_retrieval_counts(predictions, groundtruth):
    predictions, groundtruth = set(predictions), set(groundtruth)
    tp = len(predictions.intersection(groundtruth))
    fp = len(predictions) - tp
    fn = len(groundtruth) - tp
    hits1, hits = False, False
    for i, pred in enumerate(predictions):
        if pred not in groundtruth:
            continue
        hits = True
        if i == 0:
            hits1 = True
    return tp, fp, fn, hits1, hits

def calculate_retrieval_metrics(tp, fp, fn, hits1, hits, count):
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "hits@1": hits1 / count,
        "hits": hits / count
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
    print(f"Loading in LLM and tokenizer: {LLM_MODEL_NAME}...")
    before_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"Before loading in LLM: {before_mem:.2f} GB of CUDA memory used")
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        use_auth_token=llm_token,
    ).to(device)
    after_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"After loading in LLM: {after_mem:.2f} GB of CUDA memory used")
    print(f"Total LLM CUDA memory usage: {(after_mem - before_mem):.2f} GB")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME, use_fast=False,
        use_auth_token=llm_token,
    )
    #llm_tokenizer.add_special_tokens({"pad_token": LLM_PAD_TOKEN})
    #llm_model.resize_token_embeddings(len(llm_tokenizer))
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model.config.pad_token_id = llm_tokenizer.eos_token_id
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
    llm_model, llm_tokenizer, device = None, None, None#load_llm_and_tokenizer()
    data = load_data()
    database_info = load_database_info()
    print(f"Startup time: {time.time() - start_time}")
    with open(OUTPUT_FILE, "a") as output_file:
        output_file.write("\n\n~~ NEW RUN STARTING ~~\n\n")
    with torch.no_grad():
        total_tp, total_fp, total_fn, total_hits1, total_hits, total_count = await evaluate_all(data, database_info, llm_model, llm_tokenizer, device)
    metrics = calculate_retrieval_metrics(total_tp, total_fp, total_fn, total_hits1, total_hits, total_count)
    print(metrics)
    with open(RESULTS_FILE, "a") as results_file:
        results_file.write(json.dumps(metrics))

if __name__ == "__main__":
    asyncio.run(main())   
