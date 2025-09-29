# === forward 调试落盘工具 ===
import os, io, json, time, pickle, types
from collections import deque
from typing import Any, Dict, List, Tuple, Union
import torch

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _tensor_meta(t: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
        "is_leaf": bool(t.is_leaf),
        "grad_fn": type(t.grad_fn).__name__ if t.grad_fn is not None else None,
        "storage_nbytes": t.numel() * t.element_size(),
    }

def _save_tensor(t: torch.Tensor, path: str):
    # 不修改计算图；只序列化当前张量对象（含 requires_grad 标志与当前 grad）
    # PyTorch 本身不会序列化整条 autograd 历史，这里仅保存数据与基本属性
    torch.save({"tensor": t, "meta": _tensor_meta(t)}, path)

def _walk_and_save(obj: Any, root_dir: str, prefix: str) -> Any:
    """
    深拷贝式落盘：返回一个“可 pickle 的结构副本”，其中张量替换为 {__tensor_file__: ... , meta: ...}
    同时把张量本体单独存为 .pt 文件，避免巨大的单文件与重复引用问题。
    """
    if torch.is_tensor(obj):
        fname = f"{prefix}_tensor_{id(obj)}.pt"
        fpath = os.path.join(root_dir, "tensors", fname)
        _ensure_dir(os.path.dirname(fpath))
        _save_tensor(obj, fpath)
        return {"__tensor_file__": os.path.relpath(fpath, root_dir), "meta": _tensor_meta(obj)}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _walk_and_save(v, root_dir, prefix=f"{prefix}_{str(k)}") for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        seq = [_walk_and_save(v, root_dir, prefix=f"{prefix}_{i}") for i, v in enumerate(obj)]
        return type(obj)(seq)
    elif isinstance(obj, (set, frozenset)):
        seq = [_walk_and_save(v, root_dir, prefix=f"{prefix}_s")]
        return type(obj)(seq)
    else:
        # 其他自定义对象：尽量记录其 __dict__ 与 repr
        try:
            state = getattr(obj, "__dict__", None)
            if state is not None:
                return {
                    "__object_type__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    "__repr__": repr(obj),
                    "__state__": _walk_and_save(state, root_dir, prefix=f"{prefix}_state"),
                }
            else:
                return {"__object_type__": f"{obj.__class__.__module__}.{obj.__class__.__name__}", "__repr__": repr(obj)}
        except Exception as e:
            return {"__unserializable__": True, "__repr__": repr(obj), "__error__": str(e)}

def _iter_output_tensors(outputs: Any):
    if torch.is_tensor(outputs):
        yield outputs
    elif isinstance(outputs, dict):
        for v in outputs.values():
            yield from _iter_output_tensors(v)
    elif isinstance(outputs, (list, tuple, set, frozenset)):
        for v in outputs:
            yield from _iter_output_tensors(v)

def _autograd_graph_from_outputs(outputs: Any) -> Dict[str, Any]:
    """
    从 outputs 出发遍历 autograd 图，返回：
      - nodes: 每个 Function/Tensor 节点的 id、type、tensor_meta（若有）、is_tensor_node 标志
      - edges: (src_id -> dst_id)，表示 src.next_functions 中指向 dst
    说明：PyTorch 不支持把“可回放”的计算图完整序列化；这里导出的是结构化拓扑与节点信息，
          便于后续分析与可视化（例如用 graphviz 渲染 DOT）。
    """
    def node_id(x):
        return f"{type(x).__name__}:{id(x)}"

    nodes = {}
    edges: List[Tuple[str, str]] = []
    q = deque()

    # 起点：outputs 中每个 tensor 的 grad_fn，或叶子 tensor 自身
    seen = set()
    for t in _iter_output_tensors(outputs):
        start = t.grad_fn if t.grad_fn is not None else t
        if start is not None and id(start) not in seen:
            seen.add(id(start))
            q.append(start)

    while q:
        fn = q.popleft()
        nid = node_id(fn)

        if torch.is_tensor(fn):
            nodes[nid] = {
                "id": nid,
                "type": "Tensor",
                "is_tensor_node": True,
                "tensor_meta": _tensor_meta(fn),
            }
            # 叶子张量没有 next_functions，可结束
            continue

        # Autograd Function 节点
        nodes[nid] = {
            "id": nid,
            "type": type(fn).__name__,
            "is_tensor_node": False,
        }

        # 访问 next_functions（上一层依赖）
        nexts = getattr(fn, "next_functions", None)
        if nexts:
            for nxt, _ in nexts:
                if nxt is None:
                    continue
                cid = node_id(nxt)
                edges.append((cid, nid))  # 边方向：子依赖 -> 当前
                if id(nxt) not in seen:
                    seen.add(id(nxt))
                    q.append(nxt)

        # 保存该函数节点“保存的张量”（有助于定位反向中捕获的中间量）
        saved = getattr(fn, "saved_tensors", None)
        if saved:
            if "saved_tensors" not in nodes[nid]:
                nodes[nid]["saved_tensors"] = []
            for st in saved:
                st_id = node_id(st)
                nodes[nid]["saved_tensors"].append({"id": st_id, "meta": _tensor_meta(st)})
                # 将保存张量也纳入节点集（独立张量节点）
                if st_id not in nodes:
                    nodes[st_id] = {
                        "id": st_id,
                        "type": "Tensor",
                        "is_tensor_node": True,
                        "tensor_meta": _tensor_meta(st),
                    }

    return {"nodes": list(nodes.values()), "edges": edges}

def _write_dot(graph: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph Autograd {\n")
        f.write('  rankdir=LR;\n')
        for n in graph["nodes"]:
            label = n["type"]
            if n.get("is_tensor_node"):
                meta = n.get("tensor_meta", {})
                label += f'\\n{meta.get("shape", "")}\\n{meta.get("dtype","")}'
                shape = "box"
            else:
                shape = "ellipse"
            f.write(f'  "{n["id"]}" [label="{label}", shape={shape}];\n')
        for src, dst in graph["edges"]:
            f.write(f'  "{src}" -> "{dst}";\n')
        f.write("}\n")

def dump_forward_debug(
    save_root: str,
    composite_args: Any,
    composite_kwargs: Dict[str, Any],
    outputs: Any,
    tag: str = "forward"
) -> str:
    """
    将 composite_args/kwargs 的完整结构与张量数据落盘；
    基于 outputs 导出 autograd 图（DOT 与 JSON）。
    返回该次落盘目录。
    """
    ts = _timestamp()
    out_dir = os.path.join(save_root, f"{tag}_{ts}")
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "tensors"))

    # 1) 保存结构化副本（张量拆分为独立 .pt）
    args_struct = _walk_and_save(composite_args, out_dir, prefix="args")
    kwargs_struct = _walk_and_save(composite_kwargs, out_dir, prefix="kwargs")

    with open(os.path.join(out_dir, "composite_args.pkl"), "wb") as f:
        pickle.dump(args_struct, f)
    with open(os.path.join(out_dir, "composite_kwargs.pkl"), "wb") as f:
        pickle.dump(kwargs_struct, f)

    with open(os.path.join(out_dir, "composite_args.json"), "w", encoding="utf-8") as f:
        json.dump(args_struct, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "composite_kwargs.json"), "w", encoding="utf-8") as f:
        json.dump(kwargs_struct, f, ensure_ascii=False, indent=2)

    # 2) 导出 autograd 图（基于 outputs）
    try:
        graph = _autograd_graph_from_outputs(outputs)
        with open(os.path.join(out_dir, "autograd_graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        _write_dot(graph, os.path.join(out_dir, "autograd_graph.dot"))
    except Exception as e:
        with open(os.path.join(out_dir, "autograd_error.txt"), "w", encoding="utf-8") as f:
            f.write(f"Failed to export autograd graph: {e}\n")

    # 3) 记录简要文本概览
    try:
        def _brief(x):
            if torch.is_tensor(x):
                m = _tensor_meta(x)
                return f"Tensor(shape={m['shape']}, dtype={m['dtype']}, device={m['device']}, requires_grad={m['requires_grad']})"
            return type(x).__name__
        with open(os.path.join(out_dir, "overview.txt"), "w", encoding="utf-8") as f:
            f.write("=== composite_args overview ===\n")
            f.write(repr(tuple(_brief(t) for t in (composite_args if isinstance(composite_args, (list, tuple)) else (composite_args,)))) + "\n\n")
            f.write("=== composite_kwargs overview ===\n")
            f.write(repr({k: _brief(v) for k, v in (composite_kwargs or {}).items()}) + "\n\n")
    except Exception:
        pass

    return out_dir
# === 结束：forward 调试落盘工具 ===
