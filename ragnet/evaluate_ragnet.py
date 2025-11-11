import torch
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import copy
import argparse
from dotenv import load_dotenv
from itertools import chain
import json
import wandb
import time
from consts import full_system_prompt

from biencoder import BiEncoderModule
BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
load_dotenv()


def evaluate(llm_model, llm_tokenizer, dataset):
    import pdb; pdb.set_trace()


def load_llm_and_tokenizer(llm_name="meta-llama/Llama-2-7b-chat-hf"):
    # New code for loading in LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_token = os.getenv("HF_AUTH_TOKEN")
    print(f"Loading in LLM and tokenizer: {llm_name}...")
    before_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"Before loading in LLM: {before_mem:.2f} GB of CUDA memory used")
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16,
        use_auth_token=llm_token
    ).to(device)
    after_mem = torch.cuda.memory_allocated()/1024**2 / 1000
    print(f"After loading in LLM: {after_mem:.2f} GB of CUDA memory used")
    print(f"Total LLM CUDA memory usage: {(after_mem - before_mem):.2f} GB")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name, use_fast=False,
        use_auth_token=llm_token
    )
    llm_tokenizer.add_special_tokens({"pad_token": LLM_PAD_TOKEN})
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    print("LLM tokenizer successfully loaded")
    return llm_model, llm_tokenizer


def main():
    llm_model, llm_tokenizer = load_llm_and_tokenizer()
    

if __name__ == "__main__":
    main()    