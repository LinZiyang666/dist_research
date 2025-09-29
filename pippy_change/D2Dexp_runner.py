#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""
import json
from typing import Any, Dict, List, Union
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions, generate_1f1b_pipeline_actions_pro

class PartStart(nn.Module):                          # rank 0
    def __init__(self, model, end):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:end])     # 0-5
        self.rotary_emb = model.model.rotary_emb                # 共享旋转位置

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        # if not hasattr(self, '_debug_calls'):
        #     self._debug_calls = 0
        # if self._debug_calls < 20:
        #     print(f'[PartStart] call{self._debug_calls} batch={bsz} seqlen={seqlen}')
        # self._debug_calls += 1
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for idx, layer in enumerate(self.layers):
            layer_start = time.perf_counter()
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]
            # if self._debug_calls <= 5:
            #     elapsed = (time.perf_counter() - layer_start) * 1e3
            #     print(f'[PartStart] call{self._debug_calls-1} layer{idx} {elapsed:.2f}ms')

        return hidden.contiguous(), attention_mask.contiguous()


class PartMiddle(nn.Module):
    """公共基类：仅负责若干 transformer layer。"""
    def __init__(self, model, start, end):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:end])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:                       # 兼容单矩阵传递
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


class PartEnd(nn.Module):                                # rank 4：25-27 + norm + lm_head
    def __init__(self, model, start):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:])
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


def load_config(cfg:str) -> Dict[str, Any]:
    """Accepts a dict or a JSON file path."""
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = json.load(f)
    return cfg

def plan_parser(rank: int, world: int, cfg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given the pipeline config (dict or JSON path) and a rank id,
    return the stage that contains this rank, plus the immediate
    previous/next stages' rank groups.
    """
    cfg = load_config(cfg)
    if world != cfg["total_devices"]:
        raise ValueError(f"Number of devices mismatch...")
    stages: List[Dict[str, Any]] = sorted(cfg["stage_info"], key=lambda s: s["index"])
    total_stages = cfg["total_stages"]
    total_batchs = cfg["total_batchs"]
    total_stage = cfg["total_stages"]
    # Find the stage containing the rank
    stage = next((s for s in stages if rank in s.get("ranks", [])), None)
    if stage is None:
        raise ValueError(f"Rank {rank} not found in any stage's ranks.")

    idx = stage["index"]
    shard_stage = idx

    this_g = stages[idx]["ranks"]
    prev_g = stages[idx - 1]["ranks"] if idx - 1 >= 0 else None
    next_g = stages[idx + 1]["ranks"] if idx + 1 < len(stages) else None
    
    is_group = 1 if len(this_g)>1 else 0
    is_final_stage = 1 if idx == total_stage-1 else 0

    shard_from = stages[idx]["shard_layer"][0]
    shard_to = stages[idx]["shard_layer"][1]

    return is_group,is_final_stage, shard_stage, shard_from, shard_to, prev_g, this_g, next_g, total_stages, total_batchs


def plan_batch_parser(cfg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given the pipeline config (dict or JSON path) and a rank id,
    return the stage that contains this rank, plus the immediate
    previous/next stages' rank groups.
    """
    cfg = load_config(cfg)
    stages: List[Dict[str, Any]] = sorted(cfg["stage_info"], key=lambda s: s["index"])

    # Find the stage containing the rank
    # format of batch info: [[[0,1,2],[3,8,6]],...]

    #final [[[0,[0,1,2]],[1,[3,4,5,6,7,8,9,10]]..],...]
    final_batch_info = []
    final_group_info = []
    for s in stages:
        rank_group = s["ranks"]
        batch_alloc = s["batch_allocate"]
        smallest_rank = rank_group[0]
        final_group_info.append(rank_group)
        stage_l=[]
        cur_sample_index = 0
        id = 0
        for rank_index in rank_group:
            cur_sample_chunk = list(range(cur_sample_index, cur_sample_index+batch_alloc[id]))
            cur_sample_index = cur_sample_index+batch_alloc[id]
            stage_l.append([rank_index, cur_sample_chunk])
            id+=1
            

        final_batch_info.append(stage_l)

    return final_batch_info, final_group_info



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
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)


    is_group,is_final_stage, shard_stage, shard_from, shard_to, prev_g, this_g, next_g, total_stages, total_batchs = plan_parser(rank, world, args.plan_loc)

    if shard_stage == 0:
        print(f"shard_to {shard_to}")
        stage_mod = PartStart(full,shard_to)
    elif is_final_stage == 1:
        print(f"shard_from {shard_from}")
        stage_mod = PartEnd(full,shard_from)
    else:
        print(f"shard_from {shard_from}")
        print(f"shard_to {shard_to}")
        stage_mod = PartMiddle(full, shard_from, shard_to)


    if is_group == 0:
        dist.barrier()
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=shard_stage,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=prev_g, this_group=this_g, next_group=next_g)
        
    else:
        dist.barrier()
        time.sleep(shard_stage)
        dp_group = dist.new_group(ranks=this_g, backend="gloo")
        print(f"dp组 {this_g} 初始化完成")
        stage_mod.to(device)
        print("进入DDP初始化")
        #Using DDP as the data parallelism component of our frame 
        time.sleep(shard_stage)
        stage_mod = DDP(
            stage_mod,
            device_ids=None,        # CPU don' use device_ids
            process_group=dp_group, # Used for local dp
            find_unused_parameters=False,
            init_sync = False,
        )        
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=shard_stage,
                                num_stages=world, device=device,
                                group=dist.group.WORLD,  # Used for world pp 
                                prev_group=prev_g, this_group=this_g, next_group=next_g)

    
    del full                        
    import gc; gc.collect()

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
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
    
    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    
    if stage.is_first:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        return F.cross_entropy(output, target)

    # jin: we get the total_batchs from plans, but make sure args input is scynized...
    if total_batchs!= int(args.batch_size/args.microbatch_num):
        raise ValueError(f"Mbatch {total_batchs} not equal to {int(args.batch_size/args.microbatch_num)},misbatch plan's assumption")
    
    print(f"n_microbatches {args.batch_size}")
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=args.batch_size, loss_fn=loss_fn, root_pass=args.sudo_pass)
    #sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=20, loss_fn=loss_fn, root_pass=args.sudo_pass)

    # === Memory monitor: start & register containers (CPU/Gloo safe) ===

    rank_env = int(os.environ.get("RANK", str(rank)))  # 若已定义 rank 变量，可直接用
    # monitor = MemoryMonitor(
    #     log_path=f"/tmp/mem_rank{rank_env}.jsonl",
    #     interval_s=0.5,
    #     include_tensors=True,      # 若性能有感知，可改为 False
    #     top_tensors=50,            # 多列一些大对象
    #     cuda_only_tensors=False,   # 关键：CPU-only 必须 False
    # )
    # 逐容器统计（尽量多挂一些你关心的 dict）
    # try:
    #     if hasattr(stage, "fwd_cache"):
    #        # monitor.register_container("fwd_cache", stage.fwd_cache)
    #     if hasattr(stage, "bwd_cache"):
    #        # monitor.register_container("bwd_cache", stage.bwd_cache)
    # except Exception as e:
    #     print(f"[rank {rank_env}] register stage caches failed: {e}")

    # sched 内部“打包后的前向缓存”（如果存在就监控）
    # try:
    #    # monitor.register_container("big_fwd_cache", getattr(sched, "_big_fwd_cache"))
    # except Exception as e:
    #     print(f"[rank {rank_env}] register sched big cache failed: {e}")

   # monitor.start()
    # === end ===

    
    
    batch_info,group_info = plan_batch_parser(args.plan_loc)
    actions = generate_1f1b_pipeline_actions_pro(num_stages= total_stages, total_samples = args.batch_size, num_microbatches= args.microbatch_num,
                                                 group_info=group_info, batch_info=batch_info,
                                                  upstream = args.upstream)
    # actions = generate_1f1b_pipeline_actions(num_stages= total_stages, num_microbatches= 20, upstream= 1000)
    
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)            
    prev_loss = None
    
    
    for epoch in range(1):
        if stage.is_first:
            if args.train_steps is None:
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)
            dist.broadcast(steps_tensor, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, 0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(total=int(total_steps), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        try:
            for step in range(total_steps):
                step_start_time = time.time()
                opt.zero_grad(set_to_none=True)

               # with monitor.section("sched.step"):
                if stage.is_first:
                    batch = next(data_iter)
                    inp = batch["input_ids"].to(device)
                    tgt = batch["labels"].to(device)
                    dist.broadcast(tgt, src=0)
                    sched.step(inp, target=tgt)
                else:
                    tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                    dist.broadcast(tgt, src=0)
                    sched.step(None, target=tgt)

              #  with monitor.section("opt.step"):
                    opt.step()

                
                if rank == 0:
                    step_time = time.time() - step_start_time
                    tokens_processed = batch_size * block
                    tokens_per_second = tokens_processed / step_time
                    
                    pbar.set_postfix({
                        'tokens/s': f'{tokens_per_second:.0f}',
                        'step_time': f'{step_time:.2f}s',
                        'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                    })
                    pbar.update(1)
                    
                cur_loss = getattr(sched, "last_step_loss", None)
                
                if cur_loss is not None:
                    delta = (cur_loss - prev_loss) if prev_loss is not None else None

                    print(f"[rank0] step {step+1} loss {cur_loss:.4f}"
                            + ("" if prev_loss is None else (" down" if cur_loss < prev_loss else " up" if cur_loss > prev_loss else " flat")))
                    prev_loss = cur_loss
                
                
                
                dist.barrier()
            # try:
            #    # monitor.stop()
            # except Exception:
            #     pass    
            
        except Exception as e:
            # 出错时抓一份即时快照到 stdout / log
          #  snap = monitor.snapshot()
          #  print("[MEM-SNAPSHOT-ON-ERROR]", snap)
            raise
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")
            
    

    if rank == 0:
        print("\nMerging and saving model...")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()
        
        part1_state = stage_mod.state_dict()
        for key in part1_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part1_state[key]
        
        recv = [None]
        dist.broadcast_object_list(recv, src=1)
        part2_state = recv[0]
        
        for key in part2_state:
            if key.startswith("layers"):
                layer_idx = int(key.split(".")[1]) + 14
                new_key = f"model.layers.{layer_idx}" + key[key.find(".", 7):]
                merged_state[new_key] = part2_state[key]
            elif key == "norm.weight":
                merged_state["model.norm.weight"] = part2_state[key]
            elif key == "lm_head.weight":
                merged_state["lm_head.weight"] = part2_state[key]
                
        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")
    else:
        dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    #test correctness of plan_parser...
    #print(plan_parser(0, 4, "./D2Dexp_scratch.json"))