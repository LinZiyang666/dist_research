import time, json, threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Iterable, Optional
import psutil          
import torch.distributed as dist
from schedule_runtime import _mark_done, _mark_done_chunk

@dataclass
@dataclass
class TraceEvent:
    batch_id:  int
    rank:       int
    action:     str
    stage_idx:  int
    mb_idx:     int
    start_ns:   int
    end_ns:     int
    net_series: List[Tuple[int, float, float]] = field(default_factory=list)
    
    # 新增：分块编号（一级命令无分块时为 None）
    chunk:      Optional[int] = None
    # 新增：执行状态（completed / error:...）
    status:     str = "completed"
    # (ts_ns, up_mbps, down_mbps)

class Recorder:
    _NET_ACTIONS = {"SEND_F", "RECV_F", "SEND_B", "RECV_B", "ALL_REDUCE"}

    def __init__(
        self,
        rank: int,
        net_sample_interval_ms: int = 10,
        net_actions: Optional[Iterable[str]] = None,
        measure_net: bool = True,
    ):
        self.rank = rank
        self.events: List[TraceEvent] = []
        self.enabled = True
        self.measure_net = measure_net
        self.sample_interval = max(net_sample_interval_ms, 1) / 1000.0   # 秒
        self.net_actions = set(net_actions) if net_actions else self._NET_ACTIONS

    def set_enabled(self, flag: bool):
        self.enabled = flag

    @contextmanager
    def record(self, batch_id: int, action_id: int,action: str, stage_idx: int, mb_idx: int):
        if not self.enabled:
            yield
            return

        need_net = self.measure_net and action in self.net_actions
        samples: List[Tuple[int, float, float]] = []
        stop_evt = threading.Event()

        if need_net:
            # 初始累计值
            prev_ts = time.time_ns()
            prev_io = psutil.net_io_counters()
            prev_sent, prev_recv = prev_io.bytes_sent, prev_io.bytes_recv

            def sampler():
                nonlocal prev_ts, prev_sent, prev_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up_mbps   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down_mbps = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up_mbps, down_mbps))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv

            thr = threading.Thread(target=sampler, daemon=True)
            thr.start()

        start_ns = time.time_ns()
        try:
            yield
        finally:
            end_ns = time.time_ns()
            if need_net:
                stop_evt.set()
                thr.join()

            # print(f"{action} {mb_idx}计时结束，{action_id}添加表")
            _mark_done(batch_id=batch_id,action_id=action_id)
            
            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, end_ns,
                    chunk=None,
                    status="completed",
                    net_series=samples,
                )
            )

    ...
    def record_async(
        self,
        batch_id: int,
        action_id: int, 
        action: str,
        stage_idx: int,
        mb_idx: int,
        works: List[dist.Work],
        start_ns: int,
        poll_interval: float = 0.001,
        chunk_idx: Optional[int] = None,
    ):
        if not self.enabled:
            return
        
        need_net = self.measure_net and action in self.net_actions
        need_net = False
        samples, stop_evt = [], threading.Event()

        if action in ("SEND_F", "SEND_B"):

            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, start_ns,  # Use start time as end time for now
                    chunk=chunk_idx,
                    status="posted",  
                    net_series=[],
                )
            )
            return  # Exit early
        
        if need_net:
            def sampler():
                prev_ts = time.time_ns()
                io = psutil.net_io_counters()
                prev_sent, prev_recv = io.bytes_sent, io.bytes_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up, down))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv
            threading.Thread(target=sampler, daemon=True).start()

        def waiter():
            status = "completed"
            try:
                for w in works:
                    if w.is_completed():
                        continue
                    w.wait()
                end_ns = time.time_ns()
                
                if need_net:
                    stop_evt.set()
                
                # Mark done after waiting completes
                if chunk_idx is None:
                    _mark_done(batch_id=batch_id, action_id=action_id)
                else:
                    # print(f"[{dist.get_rank()}] Marking done chunk for batch={batch_id}, action={action_id}, chunk={chunk_idx}")
                    _mark_done_chunk(batch_id=batch_id, action_id=action_id, chunk_idx=chunk_idx)
                    # print(f"[{self.rank}] DONE {action} st={stage_idx} mb={mb_idx} chunk={chunk_idx} "
                    #     f"works={len(works)}")
            except Exception as e:
                status = f"error:{type(e).__name__}"
                end_ns = time.time_ns()
            finally:
                self.events.append(
                    TraceEvent(
                        batch_id, self.rank, action, stage_idx, mb_idx,
                        start_ns, end_ns,
                        chunk=chunk_idx,
                        status=status,
                        net_series=samples,
                    )
                )
        
        threading.Thread(target=waiter, daemon=True).start()


    
    def dump(self, fname: str = None):
        fname = fname or f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
