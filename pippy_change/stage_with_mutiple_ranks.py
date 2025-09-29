# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union, List, Dict
from collections import defaultdict
import time

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from pipelining_source_code.stage import PipelineStage, InputInfo, _RecvInfo, _RootArgPlaceholder, _normalize_model_output_as_tuple
from pipelining_source_code._utils import flatten_args
from pipelining_source_code._backward import stage_backward, stage_backward_input, stage_backward_weight
logger = logging.getLogger(__name__)

# ===== TAG-ADD: tag 构造工具 =====
_DIR_SHIFT, _MB_SHIFT, _SLOT_SHIFT, _SPLIT_SHIFT = 27, 17, 8, 0
_MB_BITS,  _SLOT_BITS,  _SPLIT_BITS  = 10, 9, 8
_MB_MASK,  _SLOT_MASK,  _SPLIT_MASK  = (1<<_MB_BITS)-1, (1<<_SLOT_BITS)-1, (1<<_SPLIT_BITS)-1

MOD2ID = {"text":0, "audio":1, "vision":2, "packing":3}
# New: 2-bit modality 放在更高位（28~29 位），还余 1 bit 备用（第 30 位）
_MOD_SHIFT, _MOD_BITS = 28, 2
_MOD_MASK = (1 << _MOD_BITS) - 1


def _mk_tag(direction: int, microbatch_id: int, slot_idx: int, split_idx: int, modality: int = 0) -> int:
    """31-bit tag；高 2bit 编码 modality（0..3）。若越界则回退到一致哈希（两端独立可复现）。"""
    need_fallback = (
        microbatch_id > _MB_MASK or
        slot_idx      > _SLOT_MASK or
        split_idx     > _SPLIT_MASK or
        modality      > _MOD_MASK     # 新增：检查模态是否越界
    )
    if not need_fallback:
        # 将 modality 放在最高位区间（28~29），保持你原有 4 段布局不变
        return ((modality      & _MOD_MASK)  << _MOD_SHIFT) | \
               ((direction     & 1)          << _DIR_SHIFT) | \
               ((microbatch_id & _MB_MASK)   << _MB_SHIFT)  | \
               ((slot_idx      & _SLOT_MASK) << _SLOT_SHIFT)| \
               ((split_idx     & _SPLIT_MASK)<< _SPLIT_SHIFT)

    # fallback: 简单一致哈希压 31 bit（把 modality 一并混入）
    v = (direction & 1) << 61
    v ^= (int(modality)     & 0x3)        << 59   # 新增：混入 2-bit 模态
    v ^= (int(microbatch_id)& 0xFFFFFFFF) << 30
    v ^= (int(slot_idx)     & 0x3FFFFFFF) << 10
    v ^= (int(split_idx)    & 0x3FF)
    v ^= (v >> 33); v *= 0xff51afd7ed558ccd; v &= (1<<64)-1; v ^= (v >> 33)
    return int(v & 0x7fffffff)



class PipelineStage_with_mutiple_ranks(PipelineStage):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
        prev_group: list[int] = None, 
        this_group: list[int] = None, 
        next_group: list[int] = None
    ):
        self.prev_group = prev_group          # list[int] | None
        self.this_group = this_group          # list[int]
        self.next_group = next_group          # list[int] | None
        
        self.dp_group = dist.new_group(ranks=this_group)
        self.leader   = min(this_group)

        self.is_leader = dist.get_rank() == self.leader
        self._fwd_post_recv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._last_comm_plan = {}   # key: (kind, mb_index) -> List[int] (每分块的 op 数量)
        
        self.is_print = 0
        
        super().__init__(submodule, stage_index, num_stages, device, input_args, output_args, group, dw_builder)
        
        
        
    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.prev_group is None

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.next_group is None
    
    def _shape_inference(self, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        assert args is not None

        # --- Common ---
        dp_size = dist.get_world_size(self.dp_group)
        is_multi_dp = dp_size > 1

        # === 1) Handle inputs ===
        if self.prev_group is None:
            # First stage (pure DP or "first stage + DP")
            if is_multi_dp:
                if self.is_leader:
                    meta_args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                    # Leader must also participate in the same broadcast
                    dist.broadcast_object_list([meta_args], src=self.leader, group=self.dp_group)
                    args = meta_args
                else:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
            else:
                # dp_size == 1, no collectives needed
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            # Non-first stage: receive from previous stage's leader, then DP-broadcast locally
            prev_leader = min(self.prev_group)
            if self.is_leader:
                buf = [None]
                dist.recv_object_list(buf, src=prev_leader, group=dist.group.WORLD)  # world group
                args = buf[0]
                # Safety: ensure meta in case upstream sent real tensors by mistake
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                if is_multi_dp:
                    dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
            else:
                if is_multi_dp:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
                else:
                    # dp_size == 1 and not leader cannot happen
                    raise RuntimeError("Unexpected non-leader with dp_size==1")

        # Basic sanity
        assert isinstance(args, tuple), f"shape-infer args must be tuple, got {type(args)}"

        # === 2) Cache inputs & run a dummy forward on zeros to compute outputs' shapes ===
        self.inputs_meta = args
        args = tree_map_only(torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args)

        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        outputs_meta = tuple(tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs))
        self._configure_outputs_meta(outputs_meta)

        # === 3) Send to next stage (only leader), and DP-broadcast locally for consistency ===
        if self.next_group is not None:
            next_leader = min(self.next_group)
            if self.is_leader:
                dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)  # world group

        if is_multi_dp:
            dist.broadcast_object_list([outputs_meta], src=self.leader, group=self.dp_group)

        return outputs_meta



    
    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
        rank: int, 
        dest_rank: int
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = dest_rank
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops
    
    def _compute_1d_slices(self, numel: int, num_splits: int) -> list[tuple[int, int]]:
        """
        在 1D 长度为 numel 的序列上等量切分成 <= num_splits 份。
        如果 num_splits > numel，则最多只切 numel 份（每份至少 1 个元素）。
        返回 [(start, length), ...]，按元素序覆盖整个向量。
        """
        if num_splits <= 1 or numel <= 0:
            return [(0, numel)]
        k = min(num_splits, numel)
        base = numel // k
        rem = numel % k
        slices = []
        start = 0
        for i in range(k):
            length = base + (1 if i < rem else 0)
            slices.append((start, length))
            start += length
        return slices

    def get_fwd_send_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Forward activations: 1D-flat chunking.
        生成顺序：外层 split_idx，内层轮询所有张量；同时统计每个分块的 P2POp 数量。
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        # 预计算每个张量的 1D 视图与切片表
        # …前置不变…
        plans = []  # [(slot_idx, flat, slices, dst_stages)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            if dst_stages is None:
                continue
            if not isinstance(out, torch.Tensor):
                continue
            flat = out.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices, dst_stages))  # ===== TAG-ADD =====
            slot_ctr += 1                                       # ===== TAG-ADD =====

        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices, dst_stages in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                for _dst in dst_stages:
                    if _dst is None:
                        continue
                    # 只新增 tag，不改变你的发送逻辑
                    tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                    ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                    ops_per_chunk[split_idx] += 1


        # 记录“每分块 op 数”
        self._last_comm_plan[("SEND_F", fwd_chunk_id)] = ops_per_chunk
        return ops
            
    def get_fwd_recv_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Forward activations: 1D-flat irecv into final buffers.
        生成顺序：外层 split_idx，内层轮询所有目标缓冲；同时统计每个分块的 P2POp 数量。
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        plans = []  # [(slot_idx, buf_flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for info in recv_infos:  # 保持你原来的写法，不读 arg_idx
            if not isinstance(info, _RecvInfo):
                continue
            buf = info.buffer
            if not isinstance(buf, torch.Tensor):
                continue
            buf_flat = buf.contiguous().view(-1)
            slices = self._compute_1d_slices(buf_flat.numel(), num_splits)
            plans.append((slot_ctr, buf_flat, slices))  # ===== TAG-ADD =====
            slot_ctr += 1                               # ===== TAG-ADD =====

        for split_idx in range(max(1, num_splits)):
            for slot_idx, buf_flat, slices in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = buf_flat.narrow(0, off, ln)
                tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("RECV_F", fwd_chunk_id)] = ops_per_chunk
        return ops


    
    def finish_fwd_recv(self, fwd_chunk_id: int) -> None:
        """
        完成接收后的“粘合”：把每个 tmp copy_ 到最终的 target_view。
        """
        post_list = self._fwd_post_recv.pop(fwd_chunk_id, None)
        if not post_list:
            return
        for tmp, view in post_list:
            # 这里用 copy_，目标 view 可能是非连续的窄视图，但 copy_ 支持
            with torch.no_grad():
                view.copy_(tmp)

    def get_bwd_send_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Backward grads: 1D-flat chunking.
        生成顺序：外层 split_idx，内层轮询所有梯度张量；同时统计每个分块的 P2POp 数量。
        """
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first:
            return []

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        self.fwd_cache.pop(bwd_chunk_id, None)

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        # 预计算每个梯度的 1D 视图与切片表（只发浮点/复数梯度，保持你原逻辑）
        # …前置不变…
        plans = []  # [(slot_idx, flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if grad_recv_stage is None:
                continue
            if not isinstance(grad, torch.Tensor):
                if grad is not None:
                    raise RuntimeError(
                        f"[{self.stage_index}] expecting a gradient tensor for an input "
                        f"coming from stage {grad_recv_stage}, but got {type(grad)}"
                    )
                continue
            if not (grad.is_floating_point() or torch.is_complex(grad)):
                continue
            flat = grad.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices))  # ===== TAG-ADD =====
            slot_ctr += 1                           # ===== TAG-ADD =====

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("SEND_B", bwd_chunk_id)] = ops_per_chunk
        return ops


    def get_bwd_recv_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Backward grads: 1D-flat irecv into final buffers.
        生成顺序：外层 split_idx，内层轮询所有目标缓冲；同时统计每个分块的 P2POp 数量。
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        plans = []  # [(slot_idx, buf_flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for info in recv_infos:  # 保持你原来的写法，不读 arg_idx
            if isinstance(info, _RecvInfo) and isinstance(info.buffer, torch.Tensor):
                buf_flat = info.buffer.contiguous().view(-1)
                slices = self._compute_1d_slices(buf_flat.numel(), num_splits)
                plans.append((slot_ctr, buf_flat, slices))  # ===== TAG-ADD =====
                slot_ctr += 1                               # ===== TAG-ADD =====

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, buf_flat, slices in plans:        # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = buf_flat.narrow(0, off, ln)
                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("RECV_B", bwd_chunk_id)] = ops_per_chunk
        return ops
    
    def _execute_allreduce(self):
        """
        Execute allreduce for gradients across the data parallel group.
        This should be called after all microbatches have been processed.
        """
        if not hasattr(self, 'dp_group') or self.dp_group is None:
            return
            
        # Check if the submodule is wrapped with DDP
        if isinstance(self.submod, DistributedDataParallel):
            # If using DDP, we need to manually trigger allreduce
            # since we've been using no_sync() during backward
            self._trigger_ddp_allreduce()
        else:
            # If not using DDP, manually allreduce all gradients
            self._manual_allreduce_gradients()
    
    def _trigger_ddp_allreduce(self):
        """
        Trigger DDP's allreduce mechanism manually.
        This is used when we've been accumulating gradients with no_sync().
        """
        if not isinstance(self.submod, DistributedDataParallel):
            return
            
        # Method 1: Use prepare_for_backward and trigger with a dummy backward
        # This is the approach used in the original paste.txt
        
        # Collect all output tensors from forward passes
        # Since we don't have access to the actual outputs here, 
        # we create a dummy tensor that depends on all parameters with gradients
        dummy_tensors = []
        for param in self.submod.parameters():
            if param.requires_grad and param.grad is not None:
                # Create a dummy tensor that depends on this parameter
                # Using a very small value to not affect actual gradients
                dummy = param.sum() * 0.0
                if dummy.requires_grad:
                    dummy_tensors.append(dummy)
        
        if not dummy_tensors:
            return
            
        # Prepare reducer for backward
        self.submod.reducer.prepare_for_backward(dummy_tensors)
        
        # Temporarily enable gradient synchronization
        original_sync = self.submod.require_backward_grad_sync
        self.submod.require_backward_grad_sync = True
        
        # Trigger backward to activate allreduce
        if dummy_tensors:
            # Sum all dummy tensors and backward
            total_dummy = sum(dummy_tensors)
            total_dummy.backward()
        
        # Restore original sync setting
        self.submod.require_backward_grad_sync = original_sync
    
    def _manual_allreduce_gradients(self):
        """
        Manually perform allreduce on all gradients when not using DDP.
        """
        if self.dp_group is None:
            return
            
        world_size = dist.get_world_size(self.dp_group)
        
        # Collect all parameters with gradients
        params_with_grad = []
        for param in self.submod.parameters():
            if param.grad is not None:
                params_with_grad.append(param)
        
        if not params_with_grad:
            return
        
        # Perform allreduce on each gradient
        # Can be optimized by batching small gradients together
        for param in params_with_grad:
            # Allreduce the gradient
            dist.all_reduce(
                param.grad.data,
                op=dist.ReduceOp.SUM,
                group=self.dp_group
            )
            # Average the gradient
            param.grad.data /= world_size
    
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
        pack_size: int = 1
    ):
        total_start = time.perf_counter()

        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """


        def _is_first_stage_dp(self):
            try:
                dp_size = dist.get_world_size(self.dp_group)
            except Exception:
                dp_size = 1
            return (self.prev_group is None) and (dp_size > 1)

        need_rt_bcast = _is_first_stage_dp(self)

        composite_args = None

        use_scheduler_inputs = bool(args) and pack_size > 1

        if need_rt_bcast and not use_scheduler_inputs:
            if self.is_leader:
                if not args or len(args) == 0:
                    raise RuntimeError(
                        f"[rank{dist.get_rank()}] First-stage leader got empty args at "
                        f"fwd_chunk_id={fwd_chunk_id}. Scheduler must pass root inputs to leader."
                    )
                dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
                composite_args = args
            else:
                buf = [None]
                dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                composite_args = buf[0]
        else:
            if args:
                composite_args = args
            else:
                if getattr(self, 'prev_group', None) is None:
                    composite_args = ()
                else:
                    composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        is_first_stage = getattr(self, 'prev_group', None) is None
        has_kwargs_data = kwargs and any(v is not None for v in kwargs.values())

        if composite_args is None or (len(composite_args) == 0 and not (is_first_stage and has_kwargs_data)):
            raise RuntimeError(
                f"[rank{dist.get_rank()}] Empty composite_args after dispatch at "
                f"stage={self.stage_index}, fwd_chunk_id={fwd_chunk_id}, "
                f"is_first={is_first_stage}, has_kwargs_data={has_kwargs_data}."
            )


        composite_kwargs = kwargs or {}

        if (
            pack_size > 1
            and composite_args
            and isinstance(composite_args[0], torch.Tensor)
        ):
            # if self.stage_index == 0 and fwd_chunk_id < 10:
            #     print(f"[stage0] chunk{fwd_chunk_id} composite shape={tuple(composite_args[0].shape)}")
            mb_bs = composite_args[0].shape[0] // pack_size
            args_for_val = tuple(
                (t[:mb_bs] if isinstance(t, torch.Tensor) else t)
                for t in composite_args
            )
            kwargs_for_val = {
                k: (v[:mb_bs] if isinstance(v, torch.Tensor) else v)
                for k, v in composite_kwargs.items()
            }
        else:
            args_for_val = composite_args
            kwargs_for_val = composite_kwargs

        self._validate_fwd_input(args_for_val, kwargs_for_val)

        prep_done = time.perf_counter()

        # Compute forward
        try:
            # print(
            #         "FORWARD composite_args=", tuple(
            #             (t.shape if torch.is_tensor(t) else type(t))
            #             for t in composite_args
            #         ),
            #         "; composite_kwargs=",
            #         {k: (v.shape if torch.is_tensor(v) else type(v))
            #         for k, v in composite_kwargs}
            #     )
            
            # if self.is_print == 0:
            #     self.is_print =1
            #     from tensor_debug import dump_forward_debug
            #     import os
            #     try:
            #         save_dir = dump_forward_debug(
            #             save_root=os.path.abspath("./framework_forward"),
            #             composite_args=composite_args,
            #             composite_kwargs=composite_kwargs,
            #             outputs=None,
            #             tag=f"rank{torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}"
            #         )
            #         print(f"[forward-debug] saved to: {save_dir}")
            #     except Exception as e:
            #         print(f"[forward-debug] dump failed: {e}")
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {composite_args}
            kwargs: {composite_kwargs}
            """
            raise RuntimeError(exc_msg) from e

        fwd_done = time.perf_counter()

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last:
            self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        if pack_size > 1 and isinstance(output_tuple, tuple) and len(output_tuple):
            outputs_for_val = tuple(
                t[:mb_bs] if isinstance(t, torch.Tensor) else t
                for t in output_tuple
            )
        elif pack_size > 1 and isinstance(output_tuple, torch.Tensor):
            outputs_for_val = output_tuple[:mb_bs]
        else:
            outputs_for_val = output_tuple

        self._validate_fwd_outputs(outputs_for_val)

        total_done = time.perf_counter()

        if self.stage_index == 0 and fwd_chunk_id < 40:
            rank = dist.get_rank()
            prep_ms = (prep_done - total_start) * 1e3
            fwd_ms = (fwd_done - prep_done) * 1e3
            post_ms = (total_done - fwd_done) * 1e3
            total_ms = (total_done - total_start) * 1e3
            print(f"[rank{rank}] stage{self.stage_index} chunk{fwd_chunk_id} pack{pack_size} prep={prep_ms:.2f}ms fwd={fwd_ms:.2f}ms post={post_ms:.2f}ms total={total_ms:.2f}ms")

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
        retain_graph_for_packed_mbs: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (stage_backward(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    retain_graph_for_packed_mbs=retain_graph_for_packed_mbs
                ), None)
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
            if last_backward:
                # Manually call post backward for FSDP
                def run_post_backward(fsdp_module: FSDPModule) -> None:
                    fsdp_module.set_is_last_backward(True)
                    fsdp_module.set_reshard_after_backward(True)
                    fsdp_module.set_requires_gradient_sync(True)
                    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
                    for state in fsdp_state._state_ctx.all_states:
                        if state._fsdp_param_group:
                            state._fsdp_param_group.post_backward()

                    # it would be much better if pipelining backward invoked .backward so autograd hooks
                    # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
                    # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
                    fsdp_state._root_post_backward_final_callback()

                run_post_backward(self.submod)

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups
    
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
        retain_graph_for_packed_mbs: bool = False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
             # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)

            # === NEW (Plan B): average upstream grads across the next stage's DP replicas ===
            # If the next stage has N data-parallel members (len(self.next_group) == N),
            # each sender contributed a local gradient. To keep "mean loss" semantics,
            # divide by N on the receiver before autograd.backward.
            grads_output = self._avg_next_stage_grads(grads_output)
            # === END NEW ===

            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward,
                    retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = tuple(
            (g if dst is not None and isinstance(g, torch.Tensor) else None)
            for g, dst in zip(grads_input, self.grad_send_info)
        )
        
        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
        
    
    def _avg_next_stage_grads(self, grads):
        """
        Average upstream gradients received from the *next* stage across its DP replicas.

        Why:
        - If the next stage is data-parallel over N ranks (len(self.next_group) == N),
            each rank sends its *local* gradient dL/d(out). Summing those locally on this stage
            would implement "sum loss" semantics. To keep the common "mean loss" semantics while
            keeping LR unchanged, we should divide the received gradients by N before calling
            autograd.backward on this stage.

        Behavior:
        - No-op when there is no next stage (last stage) or N <= 1.
        - Supports Tensor / list / tuple / None (None is passed through).
        - In-place scaling is used to avoid extra allocations.
        """

        fan_in = len(self.next_group) if self.next_group is not None else 1
        if grads is None or fan_in <= 1:
            return grads

        # Helper to scale a single item
        def _scale_one(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                # in-place division; safe for gradient tensors
                return x.div_(fan_in)
            return x  # for unexpected types we leave as-is

        # Compute a small debug metric before scaling (leader only) 
        pre_max = None
        try:
            if self.is_leader:
                if isinstance(grads, (list, tuple)) and len(grads) > 0 and isinstance(grads[0], torch.Tensor):
                    pre_max = grads[0].abs().max().item()
                elif isinstance(grads, torch.Tensor):
                    pre_max = grads.abs().max().item()
        except Exception:
            pre_max = None

        # Apply scaling to common containers
        if isinstance(grads, list):
            out = [_scale_one(g) for g in grads]
        elif isinstance(grads, tuple):
            out = tuple(_scale_one(g) for g in grads)
        else:
            out = _scale_one(grads)

        # Debug print after scaling (leader only)
        if self.is_leader and pre_max is not None:
            try:
                post_max = (
                    (out[0].abs().max().item() if isinstance(out, (list, tuple)) else out.abs().max().item())
                )
                # print(
                #     f"[{dist.get_rank()}] AVG upstream grads by 1/{fan_in} "
                #     f"(stage_idx={self.stage_index}) -> max|g| {pre_max:.6f} -> {post_max:.6f}"
                # )
            except Exception:
                pass

        return out
    
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple:
        grad_recv_info_list = []
        outputs_meta = self.get_outputs_meta()

        if not self.is_last:
            for idx, dst_list in act_send_info.items():
                dst = dst_list[0]
                meta = outputs_meta[idx] if idx < len(outputs_meta) else None

                if meta is None or not torch.is_tensor(meta):
                    grad_recv_info_list.append(None)
                    continue

                if not (meta.is_floating_point() or torch.is_complex(meta)):
                    grad_recv_info_list.append(None)
                    continue

                buffer = _make_tensor_from_meta(meta, self.device)
                grad_recv_info_list.append(
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst}",
                        dst,
                        buffer,
                    )
                )

        return tuple(grad_recv_info_list)
    
    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[Optional[int]]:
        grad_send_info: list[Optional[int]] = []

        def map_recv_to_send(a):
            if isinstance(a, _RecvInfo) and getattr(a, "buffer", None) is not None:
                if isinstance(a.buffer, torch.Tensor) and (
                    a.buffer.is_floating_point() or torch.is_complex(a.buffer)
                ):
                    grad_send_info.append(a.source)
                    return a.source
            grad_send_info.append(None)
            return None

        map_aggregate(args_recv_info, map_recv_to_send)
        return grad_send_info
    
    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info is None:
                return None
            elif isinstance(info, _RecvInfo):
                return info.buffer
            else:
                # Debug信息：显示stage信息和收到的类型
                stage_info = f"stage_idx={getattr(self, 'stage_index', 'unknown')}, model_type={getattr(self, 'model_type', 'unknown')}"
                

                # 如果是_RootArgPlaceholder且是第一阶段，这可能是正常的
                if hasattr(info, '__class__') and '_RootArgPlaceholder' in str(type(info)):
                    if getattr(self, 'prev_group', None) is None:  # 确实是第一阶段
                        return None

                raise AssertionError(f"Expected _RecvInfo or None but got {type(info)} at {stage_info}")

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )

class PipelineStage_Multimodality(PipelineStage_with_mutiple_ranks):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
        prev_group: list[int] = None, 
        this_group: list[int] = None, 
        next_group: list[int] = None,
        model_type: str = None,
        mm_prev_groups: Dict[str, List[int]] = None
    ):  
        self.model_type = model_type # four types: text, vision, audio, packing
        
        # forward 接收信息（按 mb、模态 → tuple[_RecvInfo,...]）
        self.mm_args_recv_info: dict[int, dict[str, tuple[_RecvInfo, ...]]] = defaultdict(dict)
        # backward 接收信息（按 mb、模态 → tuple[_RecvInfo,...]）
        self.mm_grad_recv_info: dict[int, dict[str, tuple[_RecvInfo, ...]]] = defaultdict(dict)

        # forward 未融合缓存（按 mb、模态 → tuple[Tensor,...] 或者 dict）
        self.mm_fwd_cache: dict[int, dict[str, tuple[torch.Tensor, ...]]] = defaultdict(dict)
        # backward 拆分后的梯度缓存（按 mb、模态 → tuple[Tensor,...]）
        self.mm_bwd_cache: dict[int, dict[str, tuple[torch.Tensor, ...]]] = defaultdict(dict)
        self._mm_pack_map: dict[int, dict[str, Any]] = {}  # {mb: {"order": List[Optional[tuple(mod, idx)]], "sizes": {"text":N,"audio":M,"vision":K}}}
        
        self.mm_prev_groups: Dict[str, List[int]] = mm_prev_groups or {}
        assert all(k in ["text","vision","audio","packing"] for k in self.mm_prev_groups.keys())


        
        super().__init__(submodule, stage_index, num_stages, device, input_args, output_args, group, dw_builder, prev_group,  this_group, next_group)
        
        
    def _shape_inference(self, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        assert args is not None

        # --- Common ---
        dp_size = dist.get_world_size(self.dp_group)
        is_multi_dp = dp_size > 1

        # =============== packing 分支 ===============
        if getattr(self, "model_type", None) == "packing":
            # 1) 只处理 leader 与多上游交互；组内 DP 成员由 leader 广播
            if self.is_leader:
                # ---- 1.1 从多路上游 leader 收 object meta ----
                # 约定：self.mm_prev_groups 是一个 dict: modality -> List[int] (world ranks)
                mm_prev = getattr(self, "mm_prev_groups", None)
                if not mm_prev or not isinstance(mm_prev, dict):
                    raise RuntimeError(
                        "packing stage requires self.mm_prev_groups = {'text': [...], 'audio': [...], 'vision': [...]} "
                        "with each list being the WORLD group ranks of that modality's previous stage."
                    )

                # 收齐各模态的 outputs_meta（object）
                mm_args_meta = {}
                for mod in ("text", "audio", "vision"):
                    if mod not in mm_prev or not mm_prev[mod]:
                        continue  # 该模态缺省
                    src_leader = min(mm_prev[mod])
                    buf = [None]
                    dist.recv_object_list(buf, src=src_leader, group=dist.group.WORLD)
                    # buf[0] 是上游该模态 stage 的 outputs_meta（可能是 tuple/list，含 meta tensor 和/或标量元信息）
                    mm_args_meta[mod] = buf[0]

                if "text" not in mm_args_meta:
                    # 对于我们的 Stage1 设计，positional inputs 依赖 text 的 (hidden, attn_4d, position_ids/None)
                    raise RuntimeError("packing stage expects 'text' modality upstream meta for positional inputs.")

                # ---- 1.2 设定本 stage 的 positional inputs_meta（使用 text 的三元组）----
                text_meta = mm_args_meta["text"]
                if isinstance(text_meta, (list, tuple)):
                    # 只保留 tensor 型 meta（非 tensor 项保持原状不会参与 zeros_like）
                    positional_meta = tuple(text_meta)
                else:
                    positional_meta = (text_meta,)

                # 基础 sanity：positional_meta 一定是 tuple
                assert isinstance(positional_meta, tuple), f"shape-infer args must be tuple, got {type(positional_meta)}"

                # ---- 1.3 用 zeros 跑一次子模块，得到 outputs_meta ----
                self.inputs_meta = positional_meta
                zeros_args = tree_map_only(torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), positional_meta)
                with torch.no_grad():
                    outputs = self.submod(*zeros_args, **kwargs)
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
                outputs_meta = tuple(tree_map_only(torch.Tensor, lambda x: x.to('meta'), outputs))
                self._configure_outputs_meta(outputs_meta)

                # 额外：把三路上游 meta/规划信息缓存到对象上，供后续收发/切分使用
                # - 这些对象不参与 DP 通信时的 zeros 构造，仅作通信计划依据
                self.mm_inputs_meta = mm_args_meta  # e.g., {'text': (...), 'audio': (...), 'vision': (...)}

                # ---- 1.4 把 outputs_meta 发给下游 leader（若存在），并在本 DP 组内广播 ----
                if self.next_group is not None:
                    next_leader = min(self.next_group)
                    dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)

                # 广播 positional_meta 与 outputs_meta（保证本 DP 组一致）
                if is_multi_dp:
                    dist.broadcast_object_list([positional_meta], src=self.leader, group=self.dp_group)
                    dist.broadcast_object_list([outputs_meta],   src=self.leader, group=self.dp_group)
                    # 建议再把多模态 meta（仅对象、非张量）广播一次，便于本组成员拿到规划
                    dist.broadcast_object_list([self.mm_inputs_meta], src=self.leader, group=self.dp_group)

                return outputs_meta

            else:
                # 非 leader：等待 leader 广播
                if is_multi_dp:
                    buf_pos = [None]
                    buf_out = [None]
                    buf_mm  = [None]
                    dist.broadcast_object_list(buf_pos, src=self.leader, group=self.dp_group)
                    dist.broadcast_object_list(buf_out, src=self.leader, group=self.dp_group)
                    dist.broadcast_object_list(buf_mm,  src=self.leader, group=self.dp_group)
                    self.inputs_meta   = buf_pos[0]
                    outputs_meta       = buf_out[0]
                    self.mm_inputs_meta = buf_mm[0]     # {'text': meta, 'audio': meta, 'vision': meta} 或缺省键
                    return outputs_meta
                else:
                    # dp_size == 1 且非 leader 不会发生
                    raise RuntimeError("Unexpected non-leader with dp_size==1 at packing stage")
        else: # text, vision, audio 自定义逻辑：使用真实小批输入推断，且过滤多余 kwargs
            mt = getattr(self, "model_type", None)
            allow = set()
            if mt == "audio":
                allow = {"audio_inputs"}
            elif mt == "vision":
                allow = {"vision_inputs"}
            elif mt == "text":
                allow = {"input_ids", "attention_mask"}

            clean_kwargs = {k: v for k, v in kwargs.items() if k in allow}

            # leader：设定 inputs_meta，并使用真实输入跑一次前向，以获得可靠的 meta 输出；组内广播给 DP 成员；
            if self.is_leader:

                # 设定 inputs_meta：优先从 args 中提取 tensor 生成 meta，否则用占位 meta
                try:
                    from pipelining_source_code._utils import flatten_args as _flat
                    flat_a = _flat(args)
                    metas = [t.to('meta') for t in flat_a if isinstance(t, torch.Tensor)]
                    if len(metas) == 0:
                        # 尝试从 kwargs 里拿一个 tensor 生成占位 meta
                        for v in clean_kwargs.values():
                            if isinstance(v, torch.Tensor):
                                metas.append(v.to('meta'))
                                break
                            elif isinstance(v, dict):
                                for vv in v.values():
                                    if isinstance(vv, torch.Tensor):
                                        metas.append(vv.to('meta'))
                                        break
                            if len(metas):
                                break
                    if len(metas) == 0:
                        metas = [torch.empty(1, device=self.device).to('meta')]
                    self.inputs_meta = tuple(metas)
                except Exception:
                    # 最低保真占位
                    self.inputs_meta = (torch.empty(1, device=self.device).to('meta'),)

                # Debug: dump clean_kwargs keys/shapes for audio head
                try:
                    if mt == "audio":
                        ai = (clean_kwargs or {}).get("audio_inputs", None)
                        if ai is None:
                            print(f"[rank{dist.get_rank()}] _shape_inference(audio): audio_inputs=None")
                        elif isinstance(ai, dict):
                            feat = ai.get("input_features", None)
                            fam = ai.get("feature_attention_mask", None)
                            def _s(x):
                                return (tuple(x.shape), str(x.dtype)) if isinstance(x, torch.Tensor) else type(x).__name__
                            print(f"[rank{dist.get_rank()}] _shape_inference(audio): input_features={_s(feat)} feature_attention_mask={_s(fam)}")
                        else:
                            print(f"[rank{dist.get_rank()}] _shape_inference(audio): audio_inputs type={type(ai).__name__}")
                except Exception:
                    pass

                with torch.no_grad():
                    outputs = self.submod(*args, **clean_kwargs)
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
                if outputs is None:
                    # 音频缺省：允许空输出元信息，保证流水线继续
                    if mt == "audio":
                        outputs_meta = tuple()
                        self._configure_outputs_meta(outputs_meta)
                        if self.next_group is not None:
                            next_leader = min(self.next_group)
                            dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)
                        if is_multi_dp:
                            dist.broadcast_object_list([outputs_meta], src=self.leader, group=self.dp_group)
                        return outputs_meta
                    else:
                        # 其它模态：输出缺失视为错误
                        try:
                            def _kv_shapes(d):
                                out = {}
                                for k, v in d.items():
                                    if isinstance(v, torch.Tensor):
                                        out[k] = tuple(v.shape)
                                    elif isinstance(v, dict):
                                        out[k] = {kk: (tuple(vv.shape) if isinstance(vv, torch.Tensor) else type(vv).__name__) for kk, vv in v.items()}
                                    else:
                                        out[k] = type(v).__name__
                                return out
                        except Exception:
                            pass
                        raise RuntimeError(f"[{mt}] _shape_inference: submodule returned None; please verify inputs and encoders")

                outputs_meta = tuple(tree_map_only(torch.Tensor, lambda x: x.to('meta'), outputs))
                self._configure_outputs_meta(outputs_meta)

                if self.next_group is not None:
                    next_leader = min(self.next_group)
                    dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)

                if is_multi_dp:
                    dist.broadcast_object_list([outputs_meta], src=self.leader, group=self.dp_group)
                return outputs_meta
            else:
                if is_multi_dp:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    return buf[0]
                else:
                    # dp_size == 1 且非 leader 不会发生
                    raise RuntimeError("Unexpected non-leader with dp_size==1")

    def _ensure_mm_tables(self):
        if not hasattr(self, "_mm_fwd_post_recv"):
            self._mm_fwd_post_recv = defaultdict(list)
        if not hasattr(self, "_mm_bwd_post_recv"):
            self._mm_bwd_post_recv = defaultdict(list)

    def _peer_global_rank(self, peer_rank: int) -> int:
        return peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank)

    def _recv_peer_from_info(self, info, fallback_rank: int) -> int:
        # 多源场景：优先从 recv_info 里拿源 rank（若有），否则使用传入的 fallback
        # 你现有 _RecvInfo 通常包含源 stage/rank，可按你的字段名替换：peer_rank / src_rank / src_stage_rank
        if hasattr(info, "peer_rank") and info.peer_rank is not None:
            return int(info.peer_rank)
        if hasattr(info, "src_rank") and info.src_rank is not None:
            return int(info.src_rank)
        return int(fallback_rank)

    # =============== Forward: SEND（head → packing）================
    def get_fwd_send_ops_mm(
        self,
        fwd_chunk_id: int,
        rank: int,
        dest_rank: int,         # packing 的 leader rank（或本组对端 rank）
        modality: str,
        num_splits: int = 1
    ) -> list[dist.P2POp]:
        """
        单模态 head 的 FWD 发送：本模态内 slot 从 0 计数；tag = (0, mb, slot, split)。
        output_tuple 取自 self.fwd_cache[mb]，与基类一致。
        """

        if fwd_chunk_id not in self.fwd_cache:
            return []

        output_tuple, _ = self.fwd_cache[fwd_chunk_id]
    
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_global_rank = self._peer_global_rank(dest_rank)

        # 预计算每个张量的 1D 视图与切片表
        plans = []  # [(slot_idx, flat, slices)]
        slot_ctr = 0
        for idx, out in enumerate(output_tuple):

            # 安全地获取dst_stages，处理act_send_info不存在或缺少索引的情况
            if hasattr(self, "act_send_info") and self.act_send_info and idx in self.act_send_info:
                dst_stages = self.act_send_info[idx]
            else:
                dst_stages = (dest_rank,)

            if dst_stages is None:
                continue
            if not isinstance(out, torch.Tensor):
                continue
            flat = out.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices))
            slot_ctr += 1

        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices in plans:
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx, MOD2ID[modality])
                ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))
                ops_per_chunk[split_idx] += 1

        self._last_comm_plan[("SEND_F", fwd_chunk_id, modality)] = ops_per_chunk
        return ops


    # =============== Forward: RECV（packing ← heads）================
    def get_fwd_recv_ops_mm(
        self,
        fwd_chunk_id: int,
        rank: int,
        dest_rank: int | None = None,   # 显式指定从哪个对端接收（优先使用）
        modality: str = "",
        num_splits: int = 1,
        src_rank_fallback: int | None = None,  # 兼容旧参数：未显式传 dest_rank 时作为后备
        **kwargs,
    ) -> list[dist.P2POp]:
        """
        packing 端按“单模态”生成一批 irecv；将每个 recv buffer 收到一个临时 flat 张量里，
        等全部完成后由 finish_fwd_recv_mm() 合并落到 self.mm_fwd_cache[mb][modality]。
        """
        self._ensure_mm_tables()

        # 如果本 microbatch 没有事先建立接收信息，则按 mm_inputs_meta 的结构即时构建
        recv_infos = tuple(self.mm_args_recv_info.get(fwd_chunk_id, {}).get(modality, ()))
        if not recv_infos:
            meta_obj = getattr(self, "mm_inputs_meta", {}).get(modality, None)
            built_infos: list[_RecvInfo] = []
            if meta_obj is not None:
                def _walk_and_make(x):
                    if isinstance(x, torch.Tensor):
                        buf = _make_tensor_from_meta(x, self.device)
                        built_infos.append(
                            _RecvInfo(
                                f"recv_mm_{modality}_for_{self.stage_index}",
                                (dest_rank if dest_rank is not None else (src_rank_fallback if src_rank_fallback is not None else 0)),
                                buf,
                            )
                        )
                    elif isinstance(x, (list, tuple)):
                        for e in x:
                            _walk_and_make(e)
                    elif isinstance(x, dict):
                        for e in x.values():
                            _walk_and_make(e)
                _walk_and_make(meta_obj)
            # 注册到表中
            if fwd_chunk_id not in self.mm_args_recv_info:
                self.mm_args_recv_info[fwd_chunk_id] = {}
            self.mm_args_recv_info[fwd_chunk_id][modality] = tuple(built_infos)
            recv_infos = tuple(built_infos)
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        # 计划：每个 info 分配一个“整块 flat 缓冲”，分片 irecv 到其相应窄视图
        plans = []  # [(slot_idx, tmp_full_flat, slices, peer_global_rank, shape, dtype, device)]
        slot_ctr = 0
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue
            buf = info.buffer
            if not isinstance(buf, torch.Tensor):
                continue
            # 目标形状/属性
            shape = tuple(buf.shape)
            dtype = buf.dtype
            device = buf.device
            numel = buf.numel()

            # 为该 info 分配一块完整 flat 缓冲（避免直接写最终 args；稍后统一落盘到 mm_fwd_cache）
            tmp_full_flat = torch.empty(numel, dtype=dtype, device=device)
            slices = self._compute_1d_slices(numel, num_splits)

            # 优先使用显式指定的对端 rank；否则回退到从 info 或 fallback 中获取
            if dest_rank is not None:
                peer_global_rank = self._peer_global_rank(int(dest_rank))
            else:
                peer_rank = self._recv_peer_from_info(info, (src_rank_fallback if src_rank_fallback is not None else 0))
                peer_global_rank = self._peer_global_rank(peer_rank)

            plans.append((slot_ctr, tmp_full_flat, slices, peer_global_rank, shape, dtype, device))
            # 记录到“完成后处理”
            self._mm_fwd_post_recv[(fwd_chunk_id, modality)].append((tmp_full_flat, shape, dtype, device))
            slot_ctr += 1

        for split_idx in range(max(1, num_splits)):
            for slot_idx, tmp_full_flat, slices, peer_global_rank, shape, dtype, device in plans:
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = tmp_full_flat.narrow(0, off, ln)
                tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx, MOD2ID[modality])
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))
                ops_per_chunk[split_idx] += 1

        self._last_comm_plan[("RECV_F", fwd_chunk_id, modality)] = ops_per_chunk
        try:
            print(
                f"[rank{dist.get_rank()}] get_fwd_recv_ops_mm: mb={fwd_chunk_id} mod={modality} plans={len(plans)} ops_per_chunk={ops_per_chunk}"
            )
        except Exception:
            pass
        return ops


    def finish_fwd_recv_mm(self, fwd_chunk_id: int, modality: str) -> None:
        """
        将 get_fwd_recv_ops_mm 中各 tmp_full_flat 重组为张量列表，落到 self.mm_fwd_cache[mb][modality]。
        不修改 self.input_args，pack 前由 packing 直接从 mm_fwd_cache 取数据。
        """
        self._ensure_mm_tables()
        post_list = self._mm_fwd_post_recv.pop((fwd_chunk_id, modality), None)
        if not post_list:
            try:
                print(f"[rank{dist.get_rank()}] finish_fwd_recv_mm: mb={fwd_chunk_id} modality={modality} has no post_list; mm_fwd_cache keys now: {list(self.mm_fwd_cache.get(fwd_chunk_id, {}).keys())}")
            except Exception:
                pass
            return
        tensors = []
        for tmp_full_flat, shape, dtype, device in post_list:
            tensor = tmp_full_flat.view(*shape)
            # 为packing stage接收的张量启用梯度跟踪
            if getattr(self, "model_type", None) == "packing" and tensor.is_floating_point():
                tensor = tensor.requires_grad_(True)
            tensors.append(tensor)
        self.mm_fwd_cache[fwd_chunk_id][modality] = tuple(tensors)

        # Debug: summarize what we stored for this modality
        try:
            shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in tensors]
            dtypes = [str(t.dtype) if isinstance(t, torch.Tensor) else "N/A" for t in tensors]
            reqgs  = [bool(getattr(t, 'requires_grad', False)) if isinstance(t, torch.Tensor) else False for t in tensors]
            print(
                f"[rank{dist.get_rank()}] finish_fwd_recv_mm: mb={fwd_chunk_id} modality={modality} stored {len(tensors)} tensors; "
                f"shapes={shapes}, dtypes={dtypes}, requires_grad={reqgs}"
            )
        except Exception:
            pass


    # =============== Backward: SEND（packing → heads）================
    def get_bwd_send_ops_mm(
        self,
        bwd_chunk_id: int,
        rank: int,
        dest_rank: int,         # 对应某个模态 head 的 leader
        modality: str,
        num_splits: int = 1
    ) -> list[dist.P2POp]:
        """
        packing 端把“已按模态切分好”的 grads（来自 self.mm_bwd_cache[mb][modality]）逐分片 isend。
        """
        if not self.has_backward or self.is_first:
            try:
                print(f"[rank{dist.get_rank()}] get_bwd_send_ops_mm(skip): has_backward={self.has_backward}, is_first={self.is_first}, mb={bwd_chunk_id}, mod={modality}")
            except Exception:
                pass
            return []

        # Debug: before lookup, dump keys for this mb if any
        try:
            mb_keys = list(self.mm_bwd_cache.get(bwd_chunk_id, {}).keys())
            print(f"[rank{dist.get_rank()}] get_bwd_send_ops_mm: mb={bwd_chunk_id} mod={modality} dest_rank={dest_rank}; mm_bwd_cache keys for mb: {mb_keys}")
        except Exception:
            pass

        grads_tuple = self.mm_bwd_cache.get(bwd_chunk_id, {}).get(modality, None)
        if not grads_tuple:
            # 该模态在本 mb 上可能为空（如该 batch 无音频）
            self._last_comm_plan[("SEND_B", bwd_chunk_id, modality)] = [0 for _ in range(max(1, num_splits))]
            # Extra diagnostics: check mm_fwd_cache sizes and whether forward had this modality
            try:
                fwd_mods = self.mm_fwd_cache.get(bwd_chunk_id, {})
                sizes = {k: (len(v) if isinstance(v, (list, tuple)) else 'n/a') for k, v in fwd_mods.items()}
                print(
                    f"[rank{dist.get_rank()}] get_bwd_send_ops_mm: grads_tuple EMPTY for mb={bwd_chunk_id}, mod={modality}. "
                    f"mm_fwd_cache sizes={sizes}; will send 0 ops."
                )
            except Exception:
                pass
            return []

        # Per-element diagnostics of grads_tuple
        try:
            details = []
            for i, g in enumerate(grads_tuple):
                if isinstance(g, torch.Tensor):
                    details.append((i, tuple(g.shape), str(g.dtype), bool(g.is_floating_point() or torch.is_complex(g))))
                else:
                    details.append((i, None, None, False))
            print(f"[rank{dist.get_rank()}] get_bwd_send_ops_mm: mb={bwd_chunk_id} mod={modality} grads_tuple_details={details}")
        except Exception:
            pass

        total_elements = 0
        total_bytes = 0
        valid_grads = 0
        for i, grad in enumerate(grads_tuple):
            if grad is None:
                pass
            elif not isinstance(grad, torch.Tensor):
                pass
            else:
                elements = grad.numel()
                bytes_size = elements * grad.element_size()
                total_elements += elements
                total_bytes += bytes_size
                valid_grads += 1



        peer_global_rank = self._peer_global_rank(dest_rank)
        try:
            print(
                f"[rank{dist.get_rank()}] get_bwd_send_ops_mm: mb={bwd_chunk_id} mod={modality} grads={len(grads_tuple)} -> peer_global_rank={peer_global_rank}"
            )
        except Exception:
            pass

        plans = []  # [(slot_idx, flat, slices)]
        slot_ctr = 0
        for grad in grads_tuple:
            if grad is None or not isinstance(grad, torch.Tensor):
                slot_ctr += 1
                continue
            if not (grad.is_floating_point() or torch.is_complex(grad)):
                slot_ctr += 1
                continue
            flat = grad.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices))
            max_slice_size = max(ln for _, ln in slices) if slices else 0
            max_slice_bytes = max_slice_size * grad.element_size()
            slot_ctr += 1

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices in plans:
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                chunk_bytes = chunk_view.numel() * chunk_view.element_size()

                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx, MOD2ID[modality])
                ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))
                ops_per_chunk[split_idx] += 1

        self._last_comm_plan[("SEND_B", bwd_chunk_id, modality)] = ops_per_chunk
        try:
            print(
                f"[rank{dist.get_rank()}] get_bwd_send_ops_mm: mb={bwd_chunk_id} mod={modality} ops_per_chunk={ops_per_chunk} total_ops={len(ops)}"
            )
        except Exception:
            pass
        # 选择是否在此处 pop 掉缓存；通常等三模态都发完后再清理上层字典更安全

        return ops


    # =============== Backward: RECV（heads ← packing）================
    def get_bwd_recv_ops_mm(
        self,
        bwd_chunk_id: int,
        rank: int,
        dest_rank: int | None = None,     # 显式指定从哪个对端接收（优先使用）
        modality: str = "",
        num_splits: int = 1,
        src_rank_fallback: int | None = None,  # 兼容旧参数
        **kwargs,
    ) -> list[dist.P2POp]:
    
        self._ensure_mm_tables()
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        if not recv_infos:
            self._last_comm_plan[("RECV_B", bwd_chunk_id, modality)] = [0 for _ in range(max(1, num_splits))]
            return []

        plans = []  # [(slot_idx, tmp_full_flat, slices, peer_global_rank, shape, dtype, device)]
        slot_ctr = 0
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue
            buf = info.buffer
            if not isinstance(buf, torch.Tensor):
                continue
            shape = tuple(buf.shape)
            dtype = buf.dtype
            device = buf.device
            numel = buf.numel()
            tmp_full_flat = torch.empty(numel, dtype=dtype, device=device)
            slices = self._compute_1d_slices(numel, num_splits)

            # 优先使用显式指定的对端 rank；否则回退到从 info 或 fallback 中获取
            if dest_rank is not None:
                peer_global_rank = self._peer_global_rank(int(dest_rank))
            else:
                peer_rank = self._recv_peer_from_info(info, (src_rank_fallback if src_rank_fallback is not None else 0))
                peer_global_rank = self._peer_global_rank(peer_rank)

            plans.append((slot_ctr, tmp_full_flat, slices, peer_global_rank, shape, dtype, device))
            self._mm_bwd_post_recv[(bwd_chunk_id, modality)].append((tmp_full_flat, shape, dtype, device))
            slot_ctr += 1

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, tmp_full_flat, slices, peer_global_rank, shape, dtype, device in plans:
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = tmp_full_flat.narrow(0, off, ln)
                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx, MOD2ID[modality])
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))
                ops_per_chunk[split_idx] += 1

        self._last_comm_plan[("RECV_B", bwd_chunk_id, modality)] = ops_per_chunk
        return ops
        
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
        pack_size: int = 1
    ):
        """
        对于 text/vision/audio：走父类逻辑。
        对于 packing：从 mm_fwd_cache[mb] 取三个模态的缓存，组装成 (args, kwargs) 送入子模块；
                    并记录 flat 输入到 (modality, local_idx) 的映射，供 backward 拆分用。
        """


        if getattr(self, "model_type", None) != "packing":
            # 过滤与当前模态无关的 kwargs，避免子模块收到多余参数报错
            mt = getattr(self, "model_type", None)

            allow = set()
            if mt == "audio":
                allow = {"audio_inputs"}
            elif mt == "vision":
                allow = {"vision_inputs"}
            elif mt == "text":
                allow = {"input_ids", "attention_mask"}

            # 仅保留允许的键，其余丢弃
            clean_kwargs = None
            if kwargs:
                clean_kwargs = {k: v for k, v in kwargs.items() if k in allow}


            # 音频缺省：当没有audio_inputs时，返回dummy tensor而不是空tuple
            if mt == "audio":
                ai = (clean_kwargs or {}).get("audio_inputs", None)
                if ai is None:
                    from pipelining_source_code._utils import flatten_args as _flat
                    flat_args = _flat(args)
                    flat_kwargs = _flat(clean_kwargs or {})

                    # 创建一个dummy的audio embedding tensor
                    batch_size = 1
                    audio_seq_len = 32
                    audio_dim = 512

                    dummy_audio_embeds = torch.zeros(
                        batch_size, audio_seq_len, audio_dim,
                        device=self.device, dtype=torch.float32
                    )

                    output_tuple = (dummy_audio_embeds,)
                    self.fwd_cache[fwd_chunk_id] = (output_tuple, flat_args + flat_kwargs)
                    return output_tuple

            # 在 text/audio 头部增加前后耗时与形状的调试信息
            if mt == "text":
                import time
                _t0 = time.perf_counter()
                try:

                    rid = dist.get_rank() if dist.is_initialized() else -1

                    def _summ(x, depth: int = 0):
                        if isinstance(x, torch.Tensor):
                            return f"Tensor{tuple(x.shape)}:{x.dtype}"
                        if isinstance(x, dict):
                            if depth >= 1:
                                return {k: type(v).__name__ for k, v in x.items()}
                            return {k: _summ(v, depth + 1) for k, v in x.items()}
                        if isinstance(x, (list, tuple)):
                            if depth >= 1:
                                return [type(v).__name__ for v in x]
                            return [_summ(v, depth + 1) for v in x]
                        return type(x).__name__

                    _arg_shapes = [_summ(a) for a in args]
                    _kw_shapes = {k: _summ(v) for k, v in (clean_kwargs or {}).items()}
                    print(f"[rank{rid}] [text] forward_one_chunk enter: mb={fwd_chunk_id} args={_arg_shapes} kwargs={_kw_shapes}")
                except Exception:
                    pass
                out = super().forward_one_chunk(fwd_chunk_id, args, clean_kwargs, pack_size)
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    _ms = (time.perf_counter() - _t0) * 1000.0
                    out_tup = _normalize_model_output_as_tuple(out)
                    oshapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in out_tup]
                    print(f"[rank{rid}] [text] forward_one_chunk exit: mb={fwd_chunk_id} took={_ms:.2f}ms outputs={oshapes}")
                except Exception:
                    pass
                return out

            if mt == "audio":
                import time
                _t0 = time.perf_counter()
                try:

                    rid = dist.get_rank() if dist.is_initialized() else -1

                    def _summ(x, depth: int = 0):
                        if isinstance(x, torch.Tensor):
                            return f"Tensor{tuple(x.shape)}:{x.dtype}"
                        if isinstance(x, dict):
                            if depth >= 1:
                                return {k: type(v).__name__ for k, v in x.items()}
                            return {k: _summ(v, depth + 1) for k, v in x.items()}
                        if isinstance(x, (list, tuple)):
                            if depth >= 1:
                                return [type(v).__name__ for v in x]
                            return [_summ(v, depth + 1) for v in x]
                        return type(x).__name__

                    _arg_shapes = [_summ(a) for a in args]
                    _kw_shapes = {k: _summ(v) for k, v in (clean_kwargs or {}).items()}
                    print(f"[rank{rid}] [audio] forward_one_chunk enter: mb={fwd_chunk_id} args={_arg_shapes} kwargs={_kw_shapes}")
                except Exception:
                    pass
                out = super().forward_one_chunk(fwd_chunk_id, args, clean_kwargs, pack_size)
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    _ms = (time.perf_counter() - _t0) * 1000.0
                    out_tup = _normalize_model_output_as_tuple(out)
                    oshapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in out_tup]
                    print(f"[rank{rid}] [audio] forward_one_chunk exit: mb={fwd_chunk_id} took={_ms:.2f}ms outputs={oshapes}")
                except Exception:
                    pass
                return out

            return super().forward_one_chunk(fwd_chunk_id, args, clean_kwargs, pack_size)

        # ---------- helpers（局部，无外部依赖） ----------
        def _is_float_tensor(x):
            return isinstance(x, torch.Tensor) and (x.is_floating_point() or torch.is_complex(x))

        def _is_int_like_tensor(x):
            return isinstance(x, torch.Tensor) and (x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool))

        def _parse_text_bundle(tensors: tuple[torch.Tensor, ...]):
            """返回: hidden, attn_4d, position_ids(or None), input_ids(or None), attention_mask_2d(or None)"""
            hidden = attn_4d = pos_ids = inp_ids = attn2d = None
            if not tensors:
                return hidden, attn_4d, pos_ids, inp_ids, attn2d
            if len(tensors) >= 1 and _is_float_tensor(tensors[0]): hidden = tensors[0]
            if len(tensors) >= 2 and _is_float_tensor(tensors[1]): attn_4d = tensors[1]
            idx = 2
            if len(tensors) > idx and isinstance(tensors[idx], torch.Tensor) and tensors[idx].dim() in (2, 3):
                pos_ids = tensors[idx]; idx += 1
            B = hidden.shape[0] if isinstance(hidden, torch.Tensor) and hidden.dim() >= 2 else None
            T = hidden.shape[1] if isinstance(hidden, torch.Tensor) and hidden.dim() >= 2 else None
            for j in range(idx, len(tensors)):
                t = tensors[j]
                if not isinstance(t, torch.Tensor): continue
                if _is_int_like_tensor(t) and t.dim() == 2:
                    if B is not None and T is not None and (t.shape[0] == B and t.shape[1] == T):
                        if inp_ids is None: inp_ids = t
                        elif attn2d is None: attn2d = t
            return hidden, attn_4d, pos_ids, inp_ids, attn2d

        def _parse_vision_bundle(tensors: tuple[torch.Tensor, ...]):
            """返回: image_embeds(or None), grid_thw(or None)"""
            if not tensors: return None, None
            image_embeds = None
            for t in tensors:
                if _is_float_tensor(t): image_embeds = t; break
            grid_thw = None
            for t in tensors:
                if t is image_embeds: continue
                if isinstance(t, torch.Tensor) and (not t.is_floating_point()):
                    grid_thw = t; break
            return image_embeds, grid_thw

        def _parse_audio_bundle(tensors: tuple[torch.Tensor, ...]):
            """返回: audio_embeds(or None)"""
            if not tensors: return None
            for t in tensors:
                if _is_float_tensor(t): return t
            return None
        # 以下全是packing的逻辑
        # ---------- 从模态缓存取数据 ----------
        mm = self.mm_fwd_cache.get(fwd_chunk_id, {})
        try:
            sizes_dbg = {k: (len(v) if isinstance(v, (list, tuple)) else 'n/a') for k, v in mm.items()}
            print(f"[rank{dist.get_rank()}] packing.forward_one_chunk: mb={fwd_chunk_id} mm_fwd_cache modalities={list(mm.keys())} sizes={sizes_dbg}")
        except Exception:
            pass
        text_tuple   = mm.get("text",   tuple())
        vision_tuple = mm.get("vision", tuple())
        audio_tuple  = mm.get("audio",  tuple())


        hidden, attn_4d, position_ids, input_ids, attention_mask_2d = _parse_text_bundle(text_tuple)
        if hidden is None or attn_4d is None:
            raise RuntimeError(
                f"[rank{dist.get_rank()}] packing stage missing required text tensors at mb={fwd_chunk_id}: "
                f"hidden={type(hidden)}, attn_4d={type(attn_4d)}"
            )
        image_embeds, grid_thw = _parse_vision_bundle(vision_tuple)
        audio_embeds = _parse_audio_bundle(audio_tuple)

        # ---------- 组装子模块 (args, kwargs) ----------
        # 若 position_ids 为空，为了通过输入校验，这里传空张量；Stage1 内部可选择重算
        pos_arg = position_ids if position_ids is not None else hidden.new_empty(0)
        composite_args = (hidden, attn_4d, pos_arg)
        composite_kwargs: dict[str, Any] = {}
        if input_ids is not None:          composite_kwargs["input_ids"] = input_ids
        if attention_mask_2d is not None:  composite_kwargs["attention_mask_2d"] = attention_mask_2d
        if image_embeds is not None:       composite_kwargs["image_embeds"] = image_embeds
        if audio_embeds is not None:       composite_kwargs["audio_embeds"] = audio_embeds
        if grid_thw is not None:           composite_kwargs["grid_thw"] = grid_thw

        # ---------- 验证（与父类一致） ----------
        kwargs_for_val = composite_kwargs
        args_for_val = composite_args
        if (
            pack_size > 1
            and isinstance(hidden, torch.Tensor)
            and hidden.dim() >= 1
            and hidden.shape[0] % pack_size == 0
        ):
            mb_bs = hidden.shape[0] // pack_size
            args_for_val = (
                hidden[:mb_bs],
                attn_4d,  # 你的 attn_4d 已是 [B,1,T,T]，如需裁剪可按需改
                (position_ids[:, :mb_bs, :] if (isinstance(position_ids, torch.Tensor) and position_ids.dim()==3) else position_ids)
            )
            kwargs_for_val = {}
            for k, v in composite_kwargs.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] == hidden.shape[0]:
                    kwargs_for_val[k] = v[:mb_bs]
                else:
                    kwargs_for_val[k] = v

        self._validate_fwd_input(args_for_val, kwargs_for_val)

        # Debug: summarize inputs to submodule
        try:
            def _s(x):
                return tuple(x.shape) if isinstance(x, torch.Tensor) else None
            dbg = {
                'hidden': _s(hidden), 'attn_4d': _s(attn_4d), 'position_ids': _s(position_ids),
                'input_ids': _s(input_ids), 'attention_mask_2d': _s(attention_mask_2d),
                'image_embeds': _s(image_embeds), 'grid_thw': _s(grid_thw), 'audio_embeds': _s(audio_embeds)
            }
            print(f"[rank{dist.get_rank()}] packing.forward_one_chunk: mb={fwd_chunk_id} composed inputs: {dbg}")
        except Exception:
            pass

        # ---------- 真正前向 ----------
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run packing forward:
            args: hidden={tuple(hidden.shape)}, attn_4d={tuple(attn_4d.shape)}, pos={'None' if position_ids is None else tuple(position_ids.shape)}
            kwargs: { {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)) for k,v in composite_kwargs.items()} }
            """
            raise RuntimeError(exc_msg) from e

        output_tuple = _normalize_model_output_as_tuple(output)


        if self.is_last:
            self.output_chunks.append(output)

        # ---------- 第二点：记录 flat 输入到 (modality, local_idx) 的映射（供 backward 用） ----------
        def _find_idx_in_tuple(tup, ten):
            for i, x in enumerate(tup):
                if isinstance(x, torch.Tensor) and (x is ten):
                    return i
            return None

        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        try:
            rg_flags = [bool(getattr(t, 'requires_grad', False)) if isinstance(t, torch.Tensor) else None for t in flatten_input_tensors]
            print(f"[rank{dist.get_rank()}] packing.forward_one_chunk: mb={fwd_chunk_id} flatten_input_tensors={len(flatten_input_tensors)} requires_grad={rg_flags}")
        except Exception:
            pass

        # [LEAF TENSOR FIX] 确保输入张量 requires_grad=True 且保持 leaf tensor 状态
        # 不修改输出，而是确保输入能正确追踪梯度
        for inp_tensor in flatten_input_tensors:
            if (isinstance(inp_tensor, torch.Tensor) and
                not inp_tensor.requires_grad and
                (inp_tensor.is_floating_point() or torch.is_complex(inp_tensor))):
                # 只对浮点和复数张量设置 requires_grad
                inp_tensor.requires_grad = True

        order_map: list[Optional[tuple[str, int]]] = []
        for i, ten in enumerate(flatten_input_tensors):
            if not isinstance(ten, torch.Tensor):
                order_map.append(None)
                continue
            j = _find_idx_in_tuple(text_tuple, ten)
            if j is not None:
                order_map.append(("text", j))
                continue
            j = _find_idx_in_tuple(audio_tuple, ten)
            if j is not None:
                order_map.append(("audio", j))
                continue
            j = _find_idx_in_tuple(vision_tuple, ten)
            if j is not None:
                order_map.append(("vision", j))
                continue
            order_map.append(None)  # 非 head 来源（如新建的空 pos 张量等），无需回传

        if not hasattr(self, "_mm_pack_map"):
            self._mm_pack_map = {}
        self._mm_pack_map[fwd_chunk_id] = {
            "order": order_map,
            "sizes": {
                "text":   len(text_tuple),
                "audio":  len(audio_tuple),
                "vision": len(vision_tuple),
            },
        }
        try:
            per_mod_counts = {m: sum(1 for x in order_map if isinstance(x, tuple) and x[0]==m) for m in ('text','audio','vision')}
            idxs = {m: [i for i,x in enumerate(order_map) if isinstance(x, tuple) and x[0]==m] for m in ('text','audio','vision')}
            print(f"[rank{dist.get_rank()}] packing.forward_one_chunk: mb={fwd_chunk_id} pack_map.sizes={self._mm_pack_map[fwd_chunk_id]['sizes']} mapped_counts={per_mod_counts} mapped_idxs={idxs}")
        except Exception:
            pass

        # ---------- 保存 fwd_cache，并验证输出 ----------
        # 直接保存原始的input_values，保持它们作为叶子张量
        self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)

        outputs_for_val = output_tuple
        if pack_size > 1 and isinstance(output_tuple, tuple) and len(output_tuple):
            mb_bs = hidden.shape[0] // pack_size
            outputs_for_val = tuple(
                t[:mb_bs] if isinstance(t, torch.Tensor) else t
                for t in output_tuple
            )
        self._validate_fwd_outputs(outputs_for_val)
        try:
            out_shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__ for t in (output_tuple if isinstance(output_tuple, tuple) else (output_tuple,))]
            print(f"[rank{dist.get_rank()}] packing.forward_one_chunk: mb={fwd_chunk_id} output_tuple={out_shapes}")
        except Exception:
            pass

        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
        retain_graph_for_packed_mbs: bool = False,
    ):
        """
        packing: 计算 grads_input 后，按 forward 记录的映射把梯度拆到 mm_bwd_cache[mb][modality]；
                其它模态 stage 直接用父类实现。
        """
        if getattr(self, "model_type", None) != "packing":
            # 头部 stage：沿用父类逻辑，但在 text/audio 上打印耗时与梯度统计
            mt = getattr(self, "model_type", None)
            if mt == "text":
                import time
                _t0 = time.perf_counter()
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] [text] backward_one_chunk enter: mb={bwd_chunk_id} full={full_backward} last={last_backward}")
                except Exception:
                    pass
                ret = super().backward_one_chunk(
                    bwd_chunk_id, loss, full_backward, last_backward, retain_graph_for_packed_mbs
                )
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    _ms = (time.perf_counter() - _t0) * 1000.0
                    grads_in = self.bwd_cache.get(bwd_chunk_id, ())
                    ginfo = [
                        (tuple(g.shape) if isinstance(g, torch.Tensor) else None) if g is not None else None
                        for g in grads_in
                    ]
                    # 统计参数梯度情况
                    total_params = 0
                    with_grad = 0
                    grad_norm_sum = 0.0
                    for p in self.submod.parameters():
                        total_params += 1
                        if p.grad is not None and isinstance(p.grad, torch.Tensor):
                            with_grad += 1
                            try:
                                grad_norm_sum += float(p.grad.detach().abs().mean().item())
                            except Exception:
                                pass
                    print(
                        f"[rank{rid}] [text] backward_one_chunk exit: mb={bwd_chunk_id} took={_ms:.2f}ms "
                        f"grads_in_len={len(ginfo)} shapes={ginfo} param_grads={with_grad}/{total_params} "
                        f"avg_abs_grad_sum={grad_norm_sum:.3e}"
                    )
                except Exception:
                    pass
                return ret

            if mt == "audio":
                import time
                _t0 = time.perf_counter()
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] [audio] backward_one_chunk enter: mb={bwd_chunk_id} full={full_backward} last={last_backward}")
                except Exception:
                    pass
                ret = super().backward_one_chunk(
                    bwd_chunk_id, loss, full_backward, last_backward, retain_graph_for_packed_mbs
                )
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    _ms = (time.perf_counter() - _t0) * 1000.0
                    grads_in = self.bwd_cache.get(bwd_chunk_id, ())
                    ginfo = [
                        (tuple(g.shape) if isinstance(g, torch.Tensor) else None) if g is not None else None
                        for g in grads_in
                    ]
                    # 统计参数梯度情况
                    total_params = 0
                    with_grad = 0
                    grad_norm_sum = 0.0
                    for p in self.submod.parameters():
                        total_params += 1
                        if p.grad is not None and isinstance(p.grad, torch.Tensor):
                            with_grad += 1
                            try:
                                grad_norm_sum += float(p.grad.detach().abs().mean().item())
                            except Exception:
                                pass
                    print(
                        f"[rank{rid}] [audio] backward_one_chunk exit: mb={bwd_chunk_id} took={_ms:.2f}ms "
                        f"grads_in_len={len(ginfo)} shapes={ginfo} param_grads={with_grad}/{total_params} "
                        f"avg_abs_grad_sum={grad_norm_sum:.3e}"
                    )
                except Exception:
                    pass
                return ret

            return super().backward_one_chunk(
                bwd_chunk_id, loss, full_backward, last_backward, retain_graph_for_packed_mbs
            )

        # ========== packing backward ==========
        self._check_chunk_id(bwd_chunk_id)

        # 从 forward 的 fwd_cache 拿回 stage_output 和"扁平输入列表"
        stage_output, input_values = self.fwd_cache.pop(bwd_chunk_id)


        # 1) 组装 backward 输入
        if self.is_last:
            # packing 通常不是最后一段；保留完整性
            bwd_kwargs = {"stage_output": loss, "output_grads": None, "input_values": input_values}
        else:
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            grads_output = self._avg_next_stage_grads(grads_output)
            bwd_kwargs = {"stage_output": stage_output, "output_grads": grads_output, "input_values": input_values}

        try:
            go_len = (len(bwd_kwargs["output_grads"]) if isinstance(bwd_kwargs.get("output_grads"), (list, tuple)) else None)
            print(f"[rank{dist.get_rank()}] packing.backward_one_chunk: mb={bwd_chunk_id} prep done; input_values={len(input_values)} output_grads_len={go_len}")
        except Exception:
            pass


        # 2) 真正 backward（保持与父类一致，但确保保留计算图，因同一图要给三路上游发送）
        if full_backward:
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs=True  # 关键：packing 汇聚的图别提前释放
            )
        else:
            # packing 一般不会拆成 input/weight 两段；如需支持，可参照父类写法加 input/weight 分支
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs=True
            )

        try:
            gi_types = [
                (tuple(g.shape) if isinstance(g, torch.Tensor) else None) if g is not None else None
                for g in grads_input
            ]
            print(f"[rank{dist.get_rank()}] packing.backward_one_chunk: mb={bwd_chunk_id} grads_input len={len(grads_input)} shapes={gi_types}")
            # Map grads to modalities by order
            pack_map_dbg = self._mm_pack_map.get(bwd_chunk_id, None)
            om = order if pack_map_dbg is None else pack_map_dbg.get('order', order)
            mod_to_details = {'text': [], 'audio': [], 'vision': []}
            for idx, tag in enumerate(om):
                if isinstance(tag, tuple):
                    m, local_idx = tag
                    g = grads_input[idx]
                    mod_to_details.setdefault(m, []).append((idx, local_idx, (tuple(g.shape) if isinstance(g, torch.Tensor) else None)))
            print(f"[rank{dist.get_rank()}] packing.backward_one_chunk: mb={bwd_chunk_id} grads mapping: "
                  f"text={mod_to_details.get('text')}, audio={mod_to_details.get('audio')}, vision={mod_to_details.get('vision')}")
        except Exception:
            pass


        # 3) 把 grads_input（按 flat 顺序）拆回三路模态 -> mm_bwd_cache[mb][mod]
        #    - 目标长度：与 forward 时对应模态 mm_fwd_cache[mb][mod] 的 tuple 长度一致

        pack_map = self._mm_pack_map.pop(bwd_chunk_id, None)
        if pack_map is None:
            raise RuntimeError(f"[rank{dist.get_rank()}] packing backward missing pack_map for mb={bwd_chunk_id}")

        order: list[Optional[tuple[str, int]]] = pack_map["order"]
        sizes: dict[str, int] = pack_map["sizes"]


        per_mod_lists = {
            "text":  [None] * sizes.get("text", 0),
            "audio": [None] * sizes.get("audio", 0),
            "vision":[None] * sizes.get("vision", 0),
        }

        # grads_input 的顺序与 input_values（即 forward 里 flatten_input_tensors）一致
        valid_grad_count = 0

        for grad_idx, (gi, tag) in enumerate(zip(grads_input, order)):
            if tag is None:
                continue  # 非 head 来的张量（如 position_ids 占位）或非 tensor
            mod, local_idx = tag

            if gi is None:
                pass  # Skip None gradients
            elif not isinstance(gi, torch.Tensor):
                pass  # Skip non-tensor gradients
            elif not (gi.is_floating_point() or torch.is_complex(gi)):
                pass  # Skip non-float/complex gradients
            else:
                # 只回传浮点/复数梯度，保持发送端过滤一致性
                if 0 <= local_idx < len(per_mod_lists[mod]):
                    per_mod_lists[mod][local_idx] = gi
                    valid_grad_count += 1


        # 写入 mm_bwd_cache（tuple 形式，供 get_bwd_send_ops_mm 使用）
        for mod in ("text", "audio", "vision"):
            if len(per_mod_lists[mod]) > 0:
                self.mm_bwd_cache[bwd_chunk_id][mod] = tuple(per_mod_lists[mod])

        try:
            filled_counts = {m: sum(1 for x in per_mod_lists[m] if isinstance(x, torch.Tensor)) for m in ("text","audio","vision")}
            sizes_dbg = {m: len(per_mod_lists[m]) for m in ("text","audio","vision")}
            present = list(self.mm_bwd_cache.get(bwd_chunk_id, {}).keys())
            print(f"[rank{dist.get_rank()}] packing.backward_one_chunk: mb={bwd_chunk_id} per_mod sizes={sizes_dbg} filled={filled_counts} mm_bwd_cache keys={present}")
        except Exception:
            pass

        # 4) 兼容：也把 grads_input 留在 bwd_cache，避免外部调用到基类 send 流程时出错
        if self.grad_send_info is None:
            # 与父类对齐：基于首次 args_recv_info[0] 推导；packing 实际不会用到
            try:
                self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])
            except Exception:
                self.grad_send_info = []
        self.bwd_cache[bwd_chunk_id] = grads_input

        # 5) 与父类保持相同行为：last stage detach（packing 非最后一段通常不会触发）
        if self.is_last and not self.is_first:
            for t in stage_output:
                if not t._is_view():
                    t.detach_()

        logger.debug("%s Backwarded (packing) chunk %s", self.log_prefix, bwd_chunk_id)

    
