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
from ragnet.prompts import system_prompt_lambda_dcs_examples_constrained, system_prompt_lambda_dcs_correction
from components.utils import load_json
from entity_retrieval import surface_index_memory
from eval_topk_prediction_final import denormalize_s_expr_new
from executor.logic_form_util import lisp_to_sparql
from pathlib import Path
import time
import re
from openai import AsyncOpenAI
import asyncio
import tiktoken

BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
load_dotenv()


TEST_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_test.json"
TEST_RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test_cand_rels_sorted.json"
TEST_ENTITY_DATA_NAME = "data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test_merged_cand_entities_elq_facc1.json"

TRAIN_GENERATION_DATA_NAME = "data/WebQSP/generation/merged/WebQSP_train.json"
TRAIN_RELATIONS_DATA_NAME = "data/WebQSP/relation_retrieval/candidate_relations/WebQSP_train_cand_rels_sorted.json"
TRAIN_ENTITY_DATA_NAME = "data/WebQSP/entity_retrieval/candidate_entities/WebQSP_train_merged_cand_entities_elq_facc1.json"

TRAIN_ENTITY_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_entity_label_map.json"
CANDIDATE_ENTITY_MAP_NAME = "data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_test_disamb_entities.json"
TRAIN_RELATION_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_relation_label_map.json"
TRAIN_TYPE_MAP_NAME = "data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json"

TRAIN_EMBEDDINGS_FILE = Path("ragnet/train_embeddings_all.npy") #Path("ragnet/train_embeddings_normed_expr.npy")
OUTPUT_FILE = Path("ragnet/outputs.txt")
RESULTS_FILE = Path("ragnet/results.jsonl")

LLM_MODEL_NAME = "gpt-5.2-2025-12-1" #"o3-2025-04-16" #"gpt-5.2-2025-12-11" #"gpt-5-nano" #"gpt-5.2-2025-12-11" #"gpt-5.2-2025-12-11" #"meta-llama/Llama-3.1-8B" #"meta-llama/Llama-2-7b-chat-hf"
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
    },
    "o3-2025-04-16": {
        "input": 2.00 / 1_000_000,
        "output": 8.00 / 1_000_000
    }
}

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_BATCHSIZE = 16
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13	/ 1_000_000
}

openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

BATCH_SIZE = 16

def load_data(split: str):
    data = {}
    relations_data_name = TRAIN_RELATIONS_DATA_NAME if split == "train" else TEST_RELATIONS_DATA_NAME
    entity_data_name = TRAIN_ENTITY_DATA_NAME if split == "train" else TEST_ENTITY_DATA_NAME
    generation_data_name = TRAIN_GENERATION_DATA_NAME if split == "train" else TEST_GENERATION_DATA_NAME
    with open(relations_data_name, "r") as f:
        relations_data = json.load(f)
    with open(entity_data_name, "r") as f:
        entity_data = json.load(f)
    with open(generation_data_name) as f:
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


async def evaluate_all(data, database_info, llm_model, llm_tokenizer, device, train_data, train_embeddings, batch_size=BATCH_SIZE):
    all_examples = list(data.values())
    stopping_criteria = StoppingCriteriaList([StopOnMultipleWords(["question", "q:"], llm_tokenizer)])
    all_results = []
    for i in tqdm(range(len(all_examples) // batch_size), desc="Evaluating"):
        start, end = i * batch_size, (i + 1) * batch_size
        examples_batch = all_examples[start:end]
        result = await evaluate_single(
            llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info, train_data, train_embeddings
        )
        all_results.append(result)
    tp, fp, fn, hits1, hits, count, cost = map(list, zip(*all_results))
    print(f"Total LLM cost: {sum(cost)}")
    return sum(tp), sum(fp), sum(fn), sum(hits1), sum(hits), sum(count)


class StopOnMultipleWords(StoppingCriteria):
    def __init__(self, stop_words, llm_tokenizer):
        self.stop_words = stop_words
        self.llm_tokenizer = llm_tokenizer
        self.last_n_tokens = 3

    def __call__(self, input_ids, scores, **kwargs):
        text = self.llm_tokenizer.decode(input_ids[0, -self.last_n_tokens:], skip_special_tokens=True).lower()
        return any(word.lower() in text for word in self.stop_words)


async def get_prompts(examples_batch, train_data, train_embeddings, top_k):
    prompts_batch = []
    #questions = [ example["question"] for example in examples_batch ]
    #normed_exprs = [ example["normed_sexpr"] for example in examples_batch ]
    #similarity_tasks = [ get_similar_train_examples(q, train_data, train_embeddings) for q in questions ]
    #similarity_tasks = [ get_similar_train_examples(expr, train_data, train_embeddings) for expr in normed_exprs ]
    inputs = [ example["question"] + "\n" + example["normed_sexpr"] for example in examples_batch ]
    similarity_tasks = [ get_similar_train_examples(inp, train_data, train_embeddings) for inp in inputs ]
    similarity_results = await asyncio.gather(*similarity_tasks)
    total_tokens, cost = compute_embedding_cost(inputs)
    print(f"Embedding cost: {cost}")
    for i, example in enumerate(examples_batch):
        prompt = ""
        for train_example in similarity_results[i]:
            question, question_id, entities, relations, normed_expr = train_example["question"], train_example["ID"], train_example["entities"], train_example["relations"], train_example["normed_sexpr"]
            prompt += f"\nQuestion: {question}\nEntities: {entities[:top_k]}\nRelations: {relations[:top_k]}\nLogical form: {normed_expr}"
        question, question_id, entities, relations = example["question"], example["ID"], example["entities"], example["relations"]
        prompt += f"\nQuestion: {question}\nEntities: {entities[:top_k]}\nRelations: {relations[:top_k]}\nLogical form: "
        prompts_batch.append(prompt)
    return prompts_batch


async def evaluate_single(llm_model, llm_tokenizer, device, examples_batch, stopping_criteria, database_info, train_data, train_embeddings, top_k=5):

    prompts_batch = await get_prompts(examples_batch, train_data, train_embeddings, top_k)
    all_normed_expr, all_sparql_queries, all_predictions, total_cost = await get_predictions_gpt(prompts_batch, examples_batch, database_info)
#    failed_indices = [ idx for idx in range(len(examples_batch)) if len(all_predictions[idx]) == 0 ]
#    max_retries = 2
#    retry = 1
#    while len(failed_indices) > 0 and retry < max_retries:
#        prompts_batch_failed = []
#        for idx in failed_indices:
#            prompt = prompts_batch[idx]
#            correction_prompt = f"{prompt}{all_normed_expr[idx]}\nThis last logical form is incorrect. First, provide a detailed explanation, in 1-4 sentences, explaining why it is incorrect. Then, generate a new logical form in the same format as the other logical forms, starting with Logical form: "
            #reasoning, _ = await get_response_gpt(correction_prompt, system_prompt_lambda_dcs_correction)
            #prompt_failed = f"{prompt}{all_normed_expr[idx]}\nThis last logical form is incorrect for the following reason: {reasoning}. Output the correct logical form below.\nLogical form: "
#            prompts_batch_failed.append(correction_prompt)
#        examples_batch_failed = [ examples_batch[idx] for idx in failed_indices ]
#        all_normed_expr_failed, all_sparql_queries_failed, all_predictions_failed, total_cost_failed = await get_predictions_gpt(
#            prompts_batch_failed, examples_batch_failed, database_info
#        )
#        total_cost += total_cost_failed
#        new_failed_indices = []
#        for i, idx in enumerate(failed_indices):
#            if len(all_predictions_failed[i]) == 0:
#                new_failed_indices.append(idx)
#                continue
#            all_normed_expr[idx] = all_normed_expr_failed[i]
#            all_sparql_queries[idx] = all_sparql_queries_failed[i]
#            all_predictions[idx] = all_predictions_failed[i]
#        retry += 1
#        failed_indices = new_failed_indices
        
    #get_predictions(llm_model, llm_tokenizer, device, stopping_criteria, prompts, question_id, database_info)

    # Compute evaluation metrics and save
    total_tp, total_fp, total_fn, total_hits1, total_hits, total_count = 0, 0, 0, 0, 0, 0
    for i, example in enumerate(examples_batch):
        question, question_id, entities, relations, answer = example["question"], example["ID"], example["entities"], example["relations"], example["answer"]
        gt_normed_expr, gt_sparql_query = example["normed_sexpr"], example["sparql"]
        tp, fp, fn, hits1, hits = get_retrieval_counts(all_predictions[i], answer)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_hits1 += int(hits1)
        total_hits += int(total_hits)
        total_count += 1
        with OUTPUT_FILE.open("a") as output_file:
            output = f"\n\nQuestion ID: {question_id}"
            output += f"\nPrompts batch: {prompts_batch[i]}"
            output += f"\nPredicted normed expr: {all_normed_expr[i]}"
            output += f"\nPredicted query: {all_sparql_queries[i]}"
            output += f"\nPredictions: {all_predictions[i]}"
            output += f"\nGroundtruth normed expr: {gt_normed_expr}"
            output += f"\nGroundtruth query: {gt_sparql_query}"
            output += f"\nAnswer: {answer}"
            output += f"\nTP: {total_tp}, FP: {total_fp}, FN: {total_fn}"
            output += f"\nHits@1: {hits1}, Hits: {hits}"
            output_file.write(output)
            #print(output)
    return total_tp, total_fp, total_fn, total_hits1, total_hits, total_count, total_cost


def compute_cosine_similarity(v, M):
    dot = M @ v.T
    v_norm = np.linalg.norm(v)
    M_norm = np.linalg.norm(M, axis=1, keepdims=True)
    return (dot / (M_norm * v_norm)).ravel()


async def get_similar_train_examples(test_input, train_data, train_embeddings, top_k=3):
    embed = await get_embedding(test_input)
    test_embedding = np.array(embed).reshape((1, -1))
    cosine_sim = compute_cosine_similarity(test_embedding, train_embeddings)
    top_k_indices = np.argpartition(cosine_sim, -top_k)[-top_k:]
    top_k_indices = top_k_indices[np.argsort(cosine_sim[top_k_indices])[::-1]]
    return [ list(train_data.values())[idx] for idx in top_k_indices ]


async def get_response_gpt(prompt: str, system_prompt: str):
    content, usage = None, None
    try:
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        usage = response.usage
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed GPT generation: {e}")
    return content, usage

def post_process_normed_expr_gpt(expr):
    if "Logical form" in expr: # Cut off everything before prefix
        expr = expr[expr.index("Logical form"):]
    expr = expr.replace("Logical form:", "") # Remove prefix
    expr = expr.replace('_', ' ') # remove underscores
    expr = expr.replace(".", " , ") # replace periods
    expr = expr.replace('\n', ' ').replace('\t', ' ')   # remove newlines and tabs
    expr = re.sub(r'\s+', ' ', expr).strip() # collapse multiple spaces
    return expr


async def get_predictions_gpt(prompts, examples_batch, database_info):
    all_normed_expr = [None for _ in prompts]
    all_sparql_queries = [None for _ in prompts]
    all_predictions = [[] for _ in prompts]

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = [get_response_gpt(prompt, system_prompt_lambda_dcs_examples_constrained) for prompt in prompts]
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
            question_id = examples_batch[i]["ID"]
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
    print(f"LLM cost: {total_cost}")

    return all_normed_expr, all_sparql_queries, all_predictions, total_cost

    

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


def compute_embedding_cost(texts):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    total_tokens = sum(len(enc.encode(t)) for t in texts)
    cost = (total_tokens / 1_000_000) * EMBEDDING_PRICING[EMBEDDING_MODEL_NAME]
    return total_tokens, cost


embedding_semaphore = asyncio.Semaphore(EMBEDDING_BATCHSIZE)
async def get_embedding(text_input: str):
    async with embedding_semaphore:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text_input
        )
        embedding = response.data[0].embedding
        return embedding


async def get_train_embeddings():
    train_data = load_data(split="train")
    print(f"Using train embeddings file: {TRAIN_EMBEDDINGS_FILE}")
    if TRAIN_EMBEDDINGS_FILE.exists():
        return train_data, np.load(TRAIN_EMBEDDINGS_FILE)
    #questions = [ example["question"] for example in train_data.values() ]
    #normed_exprs = [ example["normed_sexpr"] for example in train_data.values() ]
    #embedding_tasks = [ get_embedding(q) for q in questions ]
    #embedding_tasks = [ get_embedding(expr) for expr in normed_exprs ]
    inputs = [ example["question"] + "\n" + example["normed_sexpr"] for example in train_data.values() ]
    embedding_tasks = [ get_embedding(inp) for inp in inputs ]
    embeddings = await asyncio.gather(*embedding_tasks)
    train_embeddings = [ [] for _ in train_data ]
    for i, embed in enumerate(embeddings):
        train_embeddings[i] = embed
    train_embeddings = np.array(train_embeddings)
    total_tokens, cost = compute_embedding_cost(inputs)
    print(f"Embedding cost: {cost}")
    np.save(TRAIN_EMBEDDINGS_FILE, train_embeddings)
    return train_data, train_embeddings


async def main():
    start_time = time.time()
    llm_model, llm_tokenizer, device = None, None, None#load_llm_and_tokenizer()
    data = load_data(split="test")
    # TEMPORARY: splice test data
    data = {
        k: v for i, (k, v) in enumerate(data.items())
        if i < 100
    }
    database_info = load_database_info()
    print(f"Startup time: {time.time() - start_time}")
    with open(OUTPUT_FILE, "a") as output_file:
        output_file.write("\n\n~~ NEW RUN STARTING ~~\n\n")
    train_data, train_embeddings = await get_train_embeddings()
    with torch.no_grad():
        total_tp, total_fp, total_fn, total_hits1, total_hits, total_count = await evaluate_all(
            data, database_info, llm_model, llm_tokenizer, device, train_data, train_embeddings
        )
    metrics = calculate_retrieval_metrics(total_tp, total_fp, total_fn, total_hits1, total_hits, total_count)
    print(metrics)
    with open(RESULTS_FILE, "a") as results_file:
        results_file.write(f"\n{json.dumps(metrics)}")

if __name__ == "__main__":
    asyncio.run(main())   
