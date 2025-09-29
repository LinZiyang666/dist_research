##############################################
# three_rank_pp_using_original_PiPPy.py
#
# This script runs a 3-stage pipeline parallel training setup using
# the original PiPPy framework on Qwen-0.6B, for comparison against
# custom implementations such as Heterogeneous Hybrid Pipeline Parallelism.
#
# Stage Partitioning:
# ───────────────────────────────────────────
# Stage 0 : embedding + transformer layers 0–9
# Stage 1 : transformer layers 10–19
# Stage 2 : transformer layers 20–27 + normalization + lm_head
#
# Key Features:
# - Uses PiPPy's official Schedule1F1B for pipeline execution
# - Implements a clear stage-wise partition of Qwen-0.6B
# - Trains on WikiText-2 dataset using autoregressive loss
# - Measures and logs step-wise token throughput and loss
# - Merges and saves the trained model after all stages finish
#
# Intended Purpose:
# This script serves as the *baseline reference* for validating the correctness
# and performance of user-defined pipeline schedules or hybrid pipeline frameworks.
# By comparing per-step loss and training behavior with this official PiPPy implementation,
# users can identify correctness issues or efficiency gaps in custom schedules.
#
# Requirements:
# - Run with exactly 3 processes (ranks 0, 1, and 2)
# - CPU-only execution for maximum compatibility
# - Requires `transformers`, `datasets`, and `tqdm` libraries
##############################################

import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from pipelining_source_code import PipelineStage
from pipelining_source_code.schedules import ScheduleGPipe, Schedule1F1B

class Part0(nn.Module):                          # rank 0
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:10])    # 0-9
        self.rotary_emb = model.model.rotary_emb                # shared rotary

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        hidden = self.embed_tokens(input_ids)
        pos_emb = self.rotary_emb(hidden, position_ids)

        attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=device), 1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        return hidden.contiguous(), attn_mask.contiguous()


class _MiddlePart(nn.Module):
    """Base class: runs a range of transformer layers."""
    def __init__(self, model, start, end):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:end])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:  # compatibility for single matrix
            attn_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device), 1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]
        return hidden.contiguous(), attn_mask


class Part1(_MiddlePart):                              # rank 1：10-19
    def __init__(self, model):
        super().__init__(model, 10, 20)


class Part2(nn.Module):                                # rank 2：20-27 + norm + lm_head
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[20:])
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:
            attn_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device), 1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        hidden = self.norm(hidden)
        return self.lm_head(hidden)                     # logits

parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=1,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 20)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 5)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
parser.add_argument("--upstream", default=os.getenv("upstream"),
                    help='Write the upstream in mbps')
parser.add_argument("--plan_loc", type=str, required=True,
                    help='the json file that stores the sharding plans...')
args = parser.parse_args()
def main():
    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world_env = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    # Force 3 pipeline stages
    world = 3
    stage_mod = None
    if rank == 0:
        stage_mod = Part0(full)
    elif rank == 1:
        stage_mod = Part1(full)
    elif rank == 2:
        stage_mod = Part2(full)
    else:
        raise RuntimeError(f"This 3-stage script expects ranks 0..2 only, got rank={rank}")
    stage_mod.to(device)

    stage = PipelineStage(stage_mod, stage_index=rank,
                          num_stages=world, device=device,
                          group=dist.group.WORLD)
    
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
    batch_size = 100
    
    def tok_fn(ex): 
        return tok(ex["text"], return_attention_mask=False)
    
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // block * block
        ids = [concat[i:i+block] for i in range(0, tot, block)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}

    ds = (raw.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        
        bsz, seq_len, vocab_size = output.shape
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        
        return F.cross_entropy(output, target)

    # Number of microbatches should be <= batch_size
    n_microbatches = min(20, batch_size)
    sched = Schedule1F1B(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            steps = torch.tensor(len(loader), device=device)
            dist.broadcast(steps, 0)
            data_iter = iter(loader)
            #print(f"Total training steps: {steps.item()}")
        else:
            steps = torch.tensor(0, device=device)
            dist.broadcast(steps, 0)

        if rank == 0:
            pbar = tqdm(total=int(args.train_steps), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(int(args.train_steps)):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                
                # Broadcast the target shape info
                tgt_shape = torch.tensor([tgt.shape[0], tgt.shape[1]], device=device)
                dist.broadcast(tgt_shape, src=0)
                
                # Broadcast the actual target tensor
                dist.broadcast(tgt, src=0)
                
                # Step through the pipeline
                sched.step(inp, target=tgt)
            else:
                # Receive target shape
                tgt_shape = torch.empty(2, dtype=torch.long, device=device)
                dist.broadcast(tgt_shape, src=0)
                
                # Create empty target tensor with correct shape
                tgt = torch.empty(tgt_shape[0].item(), tgt_shape[1].item(), dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                
                # Step through the pipeline
                sched.step(target=tgt)
            
            opt.step()
            if rank == 0:
                tokens_processed = batch_size * block
                step_time = time.time() - step_start_time
                tokens_per_second = tokens_processed / step_time
                print("tokens:", tokens_per_second)
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)
            
                            
            cur_loss = getattr(sched, "last_step_loss", None)
            
            if cur_loss is not None:
                delta = (cur_loss - prev_loss) if prev_loss is not None else None

                #也可以根据需要额外打印一行更直观的趋势
                print(f"[rank0] step {step+1} loss {cur_loss:.4f}"
                        + ("" if prev_loss is None else (" down" if cur_loss < prev_loss else " up" if cur_loss > prev_loss else " flat")))
                prev_loss = cur_loss
            
            
            dist.barrier()
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {steps.item() / total_time:.2f} steps/s")

    if rank == 0:
        print("\nMerging and saving model.")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()

        # Rank 0 loads and saves its own part
        part0_state = stage_mod.state_dict()
        for key in part0_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part0_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part0_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part0_state[key]

        # Receive and merge state_dicts from other ranks (1 and 2)
        for i in range(1, 3):
            recv = [None]
            dist.broadcast_object_list(recv, src=i)
            part_state = recv[0]

            if i == 1:
                # Part1 only has layers 10-19
                for key in part_state:
                    if key.startswith("layers"):
                        layer_local_idx = int(key.split(".")[1])  # local 0..9
                        layer_global_idx = 10 + layer_local_idx
                        new_key = f"model.layers.{layer_global_idx}" + key[key.find('.', 7):]
                        merged_state[new_key] = part_state[key]
            else:  # i == 2 : Part2 layers 20-27 + norm + lm_head
                for key in part_state:
                    if key.startswith("layers"):
                        layer_local_idx = int(key.split(".")[1])  # local 0..7
                        layer_global_idx = 20 + layer_local_idx
                        new_key = f"model.layers.{layer_global_idx}" + key[key.find('.', 7):]
                        merged_state[new_key] = part_state[key]
                    elif key == "norm.weight":
                        merged_state["model.norm.weight"] = part_state[key]
                    elif key == "lm_head.weight":
                        merged_state["lm_head.weight"] = part_state[key]

        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")
    else:
        # Each non-root rank broadcasts its own state dict to rank 0 when requested
        dist.broadcast_object_list([stage_mod.state_dict()], src=rank)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
