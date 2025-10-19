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

from biencoder import BiEncoderModule
BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
load_dotenv()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_special_tokens', default=False, action='store_true',help='True when mask entity mention')
    parser.add_argument('--dataset_type', default="CWQ", type=str, help="CWQ | WebQSP")
    parser.add_argument('--model_save_path', default='data/', type=str)
    parser.add_argument('--max_len', default=32, type=int, help="32 for CWQ, 80 for WebQSP with richRelation, 28 for LC")
    parser.add_argument('--batch_size', default=4, type=int, help="4 for CWQ")
    parser.add_argument('--epochs', default=1, type=int, help="1 for CWQ, 3 for WebQSP")
    parser.add_argument('--log_dir', default='log/', type=str)
    parser.add_argument('--cache_dir', default='bert-base-uncased', type=str)
    args = parser.parse_args()
    return args


def data_process(dataset_type):
    if dataset_type == "CWQ":
        train_df = pd.read_csv('data/CWQ/relation_retrieval/bi-encoder/CWQ.train.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
        dev_df = pd.read_csv('data/CWQ/relation_retrieval/bi-encoder/CWQ.dev.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    else:
        # Use the model saved in last epoch
        train_df = pd.read_csv('data/WebQSP/relation_retrieval/bi-encoder/WebQSP.train.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
        dev_df = None
    
    return train_df, dev_df

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(model, device, dataloader):
    model.eval()
    
    mean_loss = 0
    count = 0
    golden_truth = []
    preds = []
    
    with torch.no_grad():
        for question_token_ids, question_attn_masks, question_token_type_ids, question, relations_token_ids, relations_attn_masks, relations_token_type_ids, relations, golden_id in tqdm(dataloader):
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            mean_loss += loss
            count += 1
            pred_id = torch.argmax(scores, dim=1) 
            # print('pred_id: {}'.format(pred_id.shape))
            # print('golden_id: {}'.format(golden_id.shape))
            preds += pred_id.tolist()
            golden_truth += golden_id.tolist()
    
    accuracy = accuracy_score(golden_truth, preds)
    
    return mean_loss / count, accuracy
    

class CustomDataset(Dataset):
    def __init__(self, data, maxlen, split, tokenizer=None, bert_model='bert-base-uncased', sample_size=100):
        self.data = data
        self.sample_size = sample_size
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen

        with open(f'data/WebQSP/generation/merged/WebQSP_{split}.json') as f: # KUNAL get normed sexpr labels
            self.generation_data = json.loads(f.read())
    
    def __len__(self):
        return int(len(self.data) / self.sample_size)
    
    def __getitem__(self, index):
        start = self.sample_size * index
        end = min(self.sample_size*(index+1), len(self.data))
        question = str(self.data.loc[start, 'question'])
        relations = [str(self.data.loc[i, 'relation']) for i in range(start, end)]
        golden_id = [i-start for i in range(start, end) if self.data.loc[i, 'label'] == 1]
        assert len(golden_id) == 1, print(start, end)
        
        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
            return_tensors='pt'
        )
        encoded_relations = [self.tokenizer(
            relation,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
            return_tensors='pt'
        ) for relation in relations]
        
        question_token_ids = encoded_question['input_ids'].squeeze(0)  # tensor of token ids
        question_attn_masks = encoded_question['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        question_token_type_ids = encoded_question['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        
        relations_token_ids = torch.cat([encoded_relation['input_ids'] for encoded_relation in encoded_relations], 0)
        relations_attn_masks = torch.cat([encoded_relation['attention_mask'] for encoded_relation in encoded_relations], 0)
        relations_token_type_ids = torch.cat([encoded_relation['token_type_ids'] for encoded_relation in encoded_relations], 0)

        normed_sexpr = [gen_data["normed_sexpr"] for gen_data in self.generation_data[start:end]] # KUNAL add
        
        return question_token_ids, question_attn_masks, question_token_type_ids, question, relations_token_ids, relations_attn_masks, relations_token_type_ids, relations, golden_id[0], normed_sexpr

LLAMA_PROMPT_FORMAT = (
"""
[INST]
<<SYS>>You are a helpful assistant that follows user instructions<</SYS>>
Question: {question}
Supporting evidence: {relations}
Answer:
[/INST]
"""
)

perplexity_system_prompt = (
"""
[INST] You are a semantic parser that converts questions into logical forms. These logical forms could be used to search a knowledge graph to obtain the answer to the question. We provide a relevant relation to help you generate the logical form.

Question: Who is Barack Obama's wife?
Relation: person.spouse.barack_obama 
Answer: ( JOIN ( R [ people , person , spouse ] ) [ Barack Obama ] )

"""
)

def calculate_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr):

    # Format the text custom
    question = [ 
       perplexity_system_prompt + f"Question: {quest}\n"
       for quest in question
    ]
    relations = np.array(relations).T # Reshape to be consistent with other tensors
    bsz, num_rels = relations.shape
    all_relations = list(chain.from_iterable(relations)) # Flatten
    for i, rel in enumerate(all_relations):
       if "|" not in rel:
          continue
       rel = rel[:rel.index("|")]
       all_relations[i] = f"Relation: {rel}\nAnswer: [/INST] "
    answers = np.array(normed_sexpr).T
    all_answers = list(chain.from_iterable(answers))

    # Get questions, relations, and answers encoded by the llm tokenizer
    encoded_question = llm_tokenizer(question, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True) # Include BOS token
    question_token_ids = encoded_question.input_ids[:, None, :].repeat((1, num_rels, 1))
    question_attn_masks = encoded_question.attention_mask[:, None, :].repeat((1, num_rels, 1))

    encoded_relations = llm_tokenizer(all_relations, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    relations_token_ids = encoded_relations.input_ids.reshape((bsz, num_rels, -1)) # Unflatten  #torch.cat([encoded.input_ids[:, None, :] for encoded in encoded_relations], dim=1)
    relations_attn_masks = encoded_relations.attention_mask.reshape((bsz, num_rels, -1)) #torch.cat([encoded.attention_mask[:, None, :] for encoded in encoded_relations], dim=-1)
    
    encoded_answer = llm_tokenizer(all_answers, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    answer_token_ids = encoded_answer.input_ids.reshape((bsz, num_rels, -1))
    answer_attn_masks = encoded_answer.attention_mask.reshape((bsz, num_rels, -1))

    # Form full sequences
    seq_token_ids = torch.cat([
        question_token_ids, relations_token_ids, answer_token_ids
    ], dim=-1)
    seq_attn_masks = torch.cat([
        question_attn_masks, relations_attn_masks, answer_attn_masks
    ], dim=-1)
    bsz, num_seqs, seq_length = seq_token_ids.shape

    # Get indices to move padding to end of sequence
    batch_indices = torch.arange(bsz)[:, None, None].repeat((1, num_seqs, seq_length))
    seq_indices = torch.arange(num_seqs)[None, :, None].repeat((bsz, 1, seq_length))
    pad_mask = (
        (seq_token_ids == llm_tokenizer.pad_token_id)
    )
    # Sort mask to move all 1s (pad) to end and 0s (nonpad) to start
    # Convert to numpy to preserve order within each category (0 vs 1)
    sorted_indices = torch.tensor(pad_mask.numpy().argsort(axis=-1, kind="stable"))

    # Move padding to end of sequence
    full_token_ids = seq_token_ids[batch_indices, seq_indices, sorted_indices]
    full_attn_masks = seq_attn_masks[batch_indices, seq_indices, sorted_indices]

    # Get mask for tokens which are part of an answer
    answer_len = answer_token_ids.size(-1)
    is_answer_mask = torch.cat([
        torch.zeros((bsz, num_seqs, seq_length - answer_len)),
        torch.ones((bsz, num_seqs, answer_len))
    ], dim=-1) * seq_attn_masks
    is_answer_mask = is_answer_mask[batch_indices, seq_indices, sorted_indices].bool()

    # Get labels
    ignore_index = -100
    full_labels = full_token_ids.masked_fill(~is_answer_mask, ignore_index)
    #TODO: get scores looking good; possibly quantize to 8bit

    # Calculate perplexity in batches
    ppl_bsz = 10
    full_perplexity = []
    for ppl_idx in range(num_seqs // ppl_bsz):

        # Get batch
        start, end = ppl_idx * ppl_bsz, (ppl_idx + 1) * ppl_bsz
        current_token_ids = full_token_ids[:, start:end, :]
        current_attn_masks = full_attn_masks[:, start:end, :]
        current_labels = full_labels[:, start:end, :]

        # Flatten batch
        flat_token_ids = current_token_ids.flatten(0, 1).to(device) #(B * N, L)
        flat_attn_masks = current_attn_masks.flatten(0, 1).to(device) # (B * N, L)
        flat_labels = current_labels.flatten(0, 1).to(device) # (B * N, L)

        # Calculate logits
        flat_logits = llm_model(
            input_ids=flat_token_ids,
            attention_mask=flat_attn_masks
        ).logits
        logits_mem = torch.cuda.memory_allocated()/1024**2 / 1000
        print(f"Memory after logits computation: {logits_mem}")

        # Shift logits and labels
        flat_logits = flat_logits[:, :-1, :] # (B * N, L-1, V)
        flat_labels = flat_labels[:, 1:] # (B * N, L-1)
        
        # Compute cross entropy loss
        flat_ce_loss = F.cross_entropy(
            flat_logits.flatten(0, 1), # (B * N * (L-1), V)
            flat_labels.flatten(0, 1), # (B * N * (L-1),)
            ignore_index=ignore_index,
            reduction="none"
        )
        #print(f"DECODED: {llm_tokenizer.decode(flat_token_ids[0])}")
        #print(f"ATTN MASKS: {flat_attn_masks}")
        #print(f"LABELS: {flat_labels}")
        #print(f"CE loss: {flat_ce_loss}")
        #print(f"PPL: {flat_ce_loss.exp()}")
        ans_idx = is_answer_mask[0, 0].float().argmax().item()
        probs = flat_logits.softmax(dim=-1)[0, ans_idx]
        preds = probs.topk(10)
        pred_tokens = llm_tokenizer.decode(preds.indices)
        #print(f"Preds: {preds}")
        #print(f"Predicted tokens: {pred_tokens}")
        ce_mem = torch.cuda.memory_allocated()/1024**2 / 1000
        print(f"Memory after CE computation: {ce_mem}")

        # Unflatten ce_loss and store it
        current_ce_loss = flat_ce_loss.reshape((bsz, ppl_bsz, seq_length - 1))
        avg_ce_loss = current_ce_loss.sum(dim=-1) / (current_ce_loss > 0).sum(dim=-1)
        current_perplexity = -avg_ce_loss.exp()
        full_perplexity.append(current_perplexity)

    # Combine all ce_loss
    perplexity = torch.cat(full_perplexity, dim=1)
    print(llm_tokenizer.decode(full_token_ids[0, 0]))
    import pdb; pdb.set_trace()
    return perplexity


def train_bert(model, llm_model, llm_tokenizer, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, model_save_path, dataset_type):
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5
    if log_path:
        log_w = open(log_path, 'w')
    scaler = GradScaler()
    best_loss = np.Inf
    best_epoch = 1
    
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        
        for it, train_batch in enumerate(tqdm(train_loader)):
            question_token_ids, question_attn_masks, question_token_type_ids, question, relations_token_ids, relations_attn_masks, relations_token_type_ids, relations, golden_id, normed_sexpr = train_batch           
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            with torch.no_grad(), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                perplexity = calculate_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr)
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        
            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()
        
            running_loss += loss.item()
            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                        .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0
        
        if val_loader:
            val_loss, accuracy = evaluate(model, device, val_loader)
            print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
            print("Accuracy on dev data: {}\n".format(accuracy))
            if log_w:
                log_w.write("Epoch {} complete! Validation Loss : {}\n".format(ep+1, val_loss))
                log_w.write("Accuracy on dev data: {}\n".format(accuracy))
        # Recording validation loss, while still saving models of every epoch
        model_copy = copy.deepcopy(model)
        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            best_loss = val_loss
            best_epoch = ep+1
        
        model_path = os.path.join(model_save_path, '{}_ep_{}.pt'.format(dataset_type, ep+1))
        torch.save(model_copy.state_dict(), model_path)
        print("The model has been saved in {}".format(model_path))

    if log_w:
        log_w.close()
    print('Best epoch is: {}, with validation loss: {}'.format(best_epoch, best_loss))
    del loss
    torch.cuda.empty_cache()
 

def main(args):
    bert_model = args.cache_dir
    freeze_bert = False
    maxlen = args.max_len
    bs = args.batch_size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = args.epochs
    log_path = os.path.join(args.log_dir, 'log.txt') 
    
    if args.add_special_tokens:
        print('add special tokens')
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(bert_model)

    set_seed(1)
    print("Reading training data...")
    train_df, dev_df = data_process(args.dataset_type)
    print(train_df.shape)
    train_set = CustomDataset(train_df, maxlen, split="train", tokenizer=tokenizer, bert_model=bert_model)
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=2)
    if dev_df is not None:
        print("Reading validation data...")
        print(dev_df.shape)
        val_set = CustomDataset(dev_df, maxlen, split="dev", tokenizer=tokenizer, bert_model=bert_model)
        val_loader = DataLoader(val_set, batch_size=bs, num_workers=2)
    else:
        val_loader = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(device, bert_model=bert_model, tokenizer=tokenizer, freeze_bert=freeze_bert)
    model.to(device)
    
    opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    
    # New code for loading in LLM
    llm_name = "meta-llama/Llama-2-7b-chat-hf"
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
    

    train_bert(model, llm_model, llm_tokenizer, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, args.model_save_path, args.dataset_type)
         

if __name__=='__main__':
    args = _parse_args()
    print(args)
    main(args)
