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


def evaluate_single(llm_model, llm_tokenizer, question, relations, top_k=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"{full_system_prompt}\nQuestion: {question}\nRelevant relations: {relations[:top_k]}\nLogical form: "
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **inputs
    )
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    import pdb; pdb.set_trace()

    
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
    llm_model, llm_tokenizer = load_llm_and_tokenizer()
    data = load_data()
    for current_id, example in data.items():
        evaluate_single(llm_model, llm_tokenizer, example["question"], example["relations"])
    

if __name__ == "__main__":
    main()    
