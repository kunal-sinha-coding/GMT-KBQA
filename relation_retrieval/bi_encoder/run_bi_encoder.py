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
from relation_retrieval.bi_encoder.consts import full_system_prompt
from ragnet.evaluate_ragnet import load_llm_and_tokenizer

from relation_retrieval.bi_encoder.biencoder import BiEncoderModule
BLANK_TOKEN = '[BLANK]'
LLM_PAD_TOKEN = '[PAD]'
MAX_RETRIES = 3
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
    parser.add_argument('--perplexity_dir', default='perplexity/', type=str)
    parser.add_argument('--start', default=0, type=float)
    parser.add_argument('--skip_loss', default=False, action='store_true', help='Save perplexity scores but dont compute loss')
    parser.add_argument('--use_baseline', default=False, action='store_true', help='Whether to run the baseline')
    args = parser.parse_args()
    return args


def data_process(dataset_type):
    if dataset_type == "CWQ":
        train_df = pd.read_csv('data/CWQ/relation_retrieval/bi_encoder/CWQ.train.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
        dev_df = pd.read_csv('data/CWQ/relation_retrieval/bi_encoder/CWQ.dev.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    else:
        # Use the model saved in last epoch
        train_df = pd.read_csv('data/WebQSP/relation_retrieval/bi_encoder/WebQSP.train.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
        dev_df = pd.read_csv('data/WebQSP/relation_retrieval/bi_encoder/WebQSP.test.sampled.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    
    return train_df, dev_df

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

DEBUG_PATH = "debug.txt"

def evaluate(model, llm_model, llm_tokenizer, device, dataloader, model_path):

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {model_path} for evaluation")
    model.eval()
    
    mean_loss = 0
    count = 0
    golden_truth = []
    preds = []
    top_relations = []

    #with open(DEBUG_PATH, "a") as f:
    #    f.write(f"\n\n\nEVALUATION:\n\n\n")
    
    with torch.no_grad():
        for it, batch in enumerate(tqdm(dataloader)):
            count += 1
            question_token_ids, question_attn_masks, question_token_type_ids, question, relations_token_ids, relations_attn_masks, relations_token_type_ids, relations, golden_id, normed_sexpr = batch
            #perplexity = get_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr, golden_id)
            scores, old_loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            #loss = calculate_replug_loss(scores, perplexity, question, relations, it, normed_sexpr)
            #wandb.log({ "test_loss": loss.item() })
            #mean_loss += loss.item()
            pred_id = torch.argmax(scores, dim=1) 
            # print('pred_id: {}'.format(pred_id.shape))
            # print('golden_id: {}'.format(golden_id.shape))
            top_relations.append(scores.argsort(dim=-1))
            preds += pred_id.tolist()
            golden_truth += golden_id.tolist()
            #relations_array = np.array(relations).T
            #for i in range(len(pred_id)):
            #    if pred_id[i].item() == golden_id[i].item():
            #        continue
            #    print(f"Question: {question[i]}")    
            #    print(f"Groundtruth LF: {normed_sexpr[i]}")
            #    print(f"Groundtruth relation: {relations_array[i, golden_id[i].item()]}")
            #    print("Predicted relations:")
            #    for j in scores[i].topk(10).indices:
            #        print(relations_array[i, j.item()])
    
    #wandb.log({ "test_loss_epoch": mean_loss / count})
    accuracy = accuracy_score(golden_truth, preds)
    wandb.log({ "test_acc": accuracy })

    # Calculate Top K metrics
    top_relations = torch.cat(top_relations, dim=0)
    golden_truth = torch.tensor(golden_truth)[:, None].to(device)
    num_relations = top_relations.size(1)
    rankings = num_relations - (top_relations == golden_truth).float().argmax(dim=-1)
    for k in [1, 2, 3, 5, 10]:
        recall = (rankings <= k).float().mean().item()
        print(f"Recall@{k}={recall}")

    return mean_loss / count, accuracy
    

class CustomDataset(Dataset):
    def __init__(self, data, maxlen, split, tokenizer=None, bert_model='bert-base-uncased', sample_size=100):
        self.data = data
        self.sample_size = sample_size
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen

        with open(f'data/WebQSP/generation/merged/WebQSP_{split}.json') as f: # KUNAL get normed sexpr labels
            self.generation_data = json.loads(f.read())
        self.generation_data_dict = {
            gen_data['question']: gen_data
            for gen_data in self.generation_data
        }
        print(f"Length of data before filtering: {len(self.data)}")
        self.data = self.data[self.data["question"].isin(self.generation_data_dict.keys())].reset_index(drop=True)
        print(f"Length of data after filtering: {len(self.data)}")
    
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

        normed_sexpr = self.generation_data_dict[question]["normed_sexpr"] if question in self.generation_data_dict else " "# KUNAL add
        
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
[INST] You are a semantic parser that converts questions into logical forms. These logical forms will be later converted into queries that can be used to search a knowledge graph and obtain the answer to the question.
As a hint, we provide a relevant relation in the knowledge graph that we will need to pass through during our search. Use this information to generate a logical form.

Question: Who is Barack Obama's wife?
Relation: person.spouse.barack_obama 
Answer: ( JOIN ( R [ person , spouse ] ) [ Barack Obama ] )

"""
)

debug_system_prompt = (
"""
[INST] What the scientific term for a dog? 
"""
)

def calculate_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr, ppl_bsz, golden_id):

    # Format the text custom
    question = [ 
       full_system_prompt + f"Question: {quest}\n"
       for quest in question
    ]
    relations_array = np.array(relations).T # Reshape to be consistent with other tensors
    bsz, num_rels = relations_array.shape
    all_relations = list(chain.from_iterable(relations_array)) # Flatten
    for i, rel in enumerate(all_relations):
       if "|" not in rel:
          continue
       rel = rel[:rel.index("|")]
       all_relations[i] = f"Relevant relation: {rel}\nLogical form: [/INST] "

    # Get questions, relations, and answers encoded by the llm tokenizer
    encoded_question = llm_tokenizer(question, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True) # Include BOS token
    question_token_ids = encoded_question.input_ids[:, None, :].repeat((1, num_rels, 1))
    question_attn_masks = encoded_question.attention_mask[:, None, :].repeat((1, num_rels, 1))

    encoded_relations = llm_tokenizer(all_relations, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    relations_token_ids = encoded_relations.input_ids.reshape((bsz, num_rels, -1)) # Unflatten  #torch.cat([encoded.input_ids[:, None, :] for encoded in encoded_relations], dim=1)
    relations_attn_masks = encoded_relations.attention_mask.reshape((bsz, num_rels, -1)) #torch.cat([encoded.attention_mask[:, None, :] for encoded in encoded_relations], dim=-1)
    
    #normed_sexpr = [relations[0, 30]] + list(normed_sexpr)[1:]
    encoded_answer = llm_tokenizer(normed_sexpr, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    answer_token_ids = encoded_answer.input_ids[:, None, :].repeat((1, num_rels, 1))
    answer_attn_masks = encoded_answer.attention_mask[:, None, :].repeat((1, num_rels, 1))

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

        # Shift logits and labels
        flat_logits = flat_logits[:, :-1, :] # (B * N, L-1, V)
        flat_labels = flat_labels[:, 1:] # (B * N, L-1)
        
        # Compute cross entropy loss
        torch.cuda.empty_cache()
        #ce_mem = torch.cuda.mem_get_info()[0] /1024**2 / 1000
        #print(f"Memory before CE computation: {ce_mem}")
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
        #ans_idx = is_answer_mask[0, 0].float().argmax().item()
        #probs = flat_logits.softmax(dim=-1)[0, ans_idx]
        #preds = flat_logits[0, ans_idx].topk(10)
        #pred_tokens = llm_tokenizer.decode(preds.indices)
        #print(f"Preds: {preds}")
        #print(f"Predicted tokens: {pred_tokens}")

        # Unflatten ce_loss and store it
        current_ce_loss = flat_ce_loss.reshape((bsz, ppl_bsz, seq_length - 1))
        avg_ce_loss = current_ce_loss.sum(dim=-1) / (current_ce_loss > 0).sum(dim=-1)
        current_perplexity = -avg_ce_loss.exp()
        full_perplexity.append(current_perplexity)

    # Combine all ce_loss
    perplexity = torch.cat(full_perplexity, dim=1)
    return perplexity

def calculate_replug_loss(scores, perplexity, question, relations, it=None, normed_sexpr=None, gamma=10):
    relations_array = np.array(relations).T
    scores_probs = scores.log_softmax(dim=-1)
    perplexity_probs = (perplexity * gamma).log_softmax(dim=-1)
    with open(DEBUG_PATH, "a") as f:
        f.write(f"Question for iteration{it}: {question}\n\n")
        for i in range(len(relations_array)):
            predicted_relations = [relations_array[i, j.item()] for j in scores_probs[i].topk(5).indices]
            perplexity_relations = [relations_array[i, j.item()] for j in perplexity_probs[i].topk(5).indices]
            f.write(f"Predicted relations: {predicted_relations}\n")
            f.write(f"Perplexity relations: {perplexity_relations}\n")
            f.write(f"Normed sexpr: {normed_sexpr}\n\n")
        kl_div = F.kl_div(scores_probs, perplexity_probs, log_target=True, reduction='none').sum(dim=-1)
        f.write(f"Loss: {kl_div}\n\n")
    return kl_div.mean(dim=0)

def get_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr, golden_id, it=0, skip_loss=False, perplexity_dir=None):
    bsz = len(question)
    start, end = bsz * it, bsz * (it + 1)
    perplexity_paths = [ 
        os.path.join(perplexity_dir, f"{i}.pt") 
        for i in range(start, end)
        if perplexity_dir
    ]
    if perplexity_dir and all([os.path.exists(path) for path in perplexity_paths]):
        if skip_loss:
            return None
        perplexity = torch.stack([torch.load(path) for path in perplexity_paths], dim=0)
    else:
        with torch.no_grad(), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ppl_bsz = 10
            for retry in range(MAX_RETRIES):
                try:
                    perplexity = calculate_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr, ppl_bsz, golden_id)
                    break
                except Exception as error:
                    print(f"ERROR: {str(error)}")
                    torch.cuda.empty_cache()
                    time.sleep(retry * 1)
                    ppl_bsz /= 2
                    continue
        for i, path in enumerate(perplexity_paths):
            torch.save(perplexity[i], path)
    return perplexity

def train_bert(
    model, llm_model, llm_tokenizer, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, 
    log_path, model_save_path, dataset_type, perplexity_dir, start, skip_loss, use_baseline
):
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5
    if log_path:
        log_w = open(log_path, 'w')
    scaler = GradScaler()
    best_loss = np.Inf
    best_epoch = 1

    #best_loss, best_epoch = run_evaluation(model, llm_model, llm_tokenizer, device, val_loader, dataset_type, model_path, 0, log_w, best_loss, best_epoch)
    
    for ep in range(1, epochs + 1):
        model.train()
        loss = 0.0
        running_loss = 0.0
        mean_loss = 0.0
        count = 0
        model_path = os.path.join(model_save_path, '{}_ep_{}_{}.pt'.format(dataset_type, ep, "baseline" if use_baseline else ""))
        import pdb; pdb.set_trace()
        perplexity = None

        for it, train_batch in enumerate(tqdm(train_loader)):
            if it < int(len(train_loader) * start):
                continue
            question_token_ids, question_attn_masks, question_token_type_ids, question, relations_token_ids, relations_attn_masks, relations_token_type_ids, relations, golden_id, normed_sexpr = train_batch           
            if not use_baseline:
                perplexity = get_perplexity(llm_model, llm_tokenizer, question, relations, device, normed_sexpr, golden_id, it, skip_loss, perplexity_dir)
            if skip_loss:
                continue
            count += 1
            scores, old_loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            if use_baseline:
                loss = old_loss
            else:
                loss = calculate_replug_loss(scores, perplexity, question, relations, it=it, normed_sexpr=normed_sexpr)
            loss = loss / iters_to_accumulate
            wandb.log({ "train_loss": loss.item() })
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
            mean_loss += loss.item()

        save_model(model, model_path)
        if count > 0:
            wandb.log({"train_loss_epoch": mean_loss / count})
        best_loss, best_epoch = run_evaluation(model, llm_model, llm_tokenizer, device, val_loader, dataset_type, model_path, ep, log_w, best_loss, best_epoch)

    if log_w:
        log_w.close()
    print('Best epoch is: {}, with validation loss: {}'.format(best_epoch, best_loss))
    del loss
    torch.cuda.empty_cache()


def save_model(model, model_path):
    model_copy = copy.deepcopy(model)
    torch.save(model_copy.state_dict(), model_path)
    print("The model has been saved in {}".format(model_path))


def run_evaluation(model, llm_model, llm_tokenizer, device, val_loader, dataset_type, model_path, ep, log_w, best_loss, best_epoch):
    if val_loader:
        val_loss, accuracy = evaluate(model, llm_model, llm_tokenizer, device, val_loader, model_path)
        print("Epoch {} complete! Validation Loss : {}".format(ep, val_loss))
        print("Accuracy on dev data: {}\n".format(accuracy))
        if log_w:
            log_w.write("Epoch {} complete! Validation Loss : {}\n".format(ep, val_loss))
            log_w.write("Accuracy on dev data: {}\n".format(accuracy))
    # Recording validation loss, while still saving models of every epoch
    if val_loss < best_loss:
        print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
        print()
        best_loss = val_loss
        best_epoch = ep
    return best_loss, best_epoch

 

def main(args):
    print("Starting wandb...")
    wandb_run = wandb.init(project="ragnet")
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
        # KUNAL edit - make this test not dev
        val_set = CustomDataset(dev_df, maxlen, split="test", tokenizer=tokenizer, bert_model=bert_model)
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
    llm_model, llm_tokenizer = load_llm_and_tokenizer()
    with open(DEBUG_PATH, "a") as f:
        f.write(f"\n\n\nEXPERIMENT: {wandb_run.name}-{wandb_run.id}\n\n\n")
    
    train_bert(
        model, llm_model, llm_tokenizer, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, 
        device, log_path, args.model_save_path, args.dataset_type, args.perplexity_dir, args.start, args.skip_loss, args.use_baseline
    )
         

if __name__=='__main__':
    args = _parse_args()
    print(args)
    main(args)
