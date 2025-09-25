#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_attr_scores_eap.py
────────────────────────────────────────────────────────────────────────────
Replace the mask_gradient_prune_scores pipeline with EAP (Edge Attribution Patching),
fully removing the auto_circuit dependency while preserving the original
dataset/split/threshold and on-disk layout.

Outputs (same path format as before):
- Sparse-edge JSON:
  outputs/attr_scores/<dataset>/<model>/<keep%>/<run_tag>/<logic_name>_split{seed}_part{A|B}.json
- Heatmap PNG: .../figures/<logic_name>_{subset_tag}.png
- Circuit PNG: .../figures/<logic_name>_{subset_tag}_circuit_top100.png
"""

from __future__ import annotations

# ───────────────────────────── Imports ─────────────────────────────────────
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pdb

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import re

# Local deps expected in your repo (same directory or on PYTHONPATH)
# - src/eap_wrapper.py: exposes EAP(...)
# - src/eap_graph.py:   exposes EAPGraph
try:
    from src.eap_wrapper import EAP  # type: ignore
    from src.eap_graph import EAPGraph  # type: ignore
except Exception as _e:
    EAP = None
    EAPGraph = None
    _IMPORT_ERR = _e

# TransformerLens HookedTransformer
from transformer_lens import HookedTransformer


# ───────────────────────────── Constants ───────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA = "data/corrupt/level_1.json"
DEFAULT_OUT  = "outputs/attr_scores"
TOP_N_EDGES  = 200          # top-N edges for circuit visualization
FIG_DPI      = 200
ATTN_EDGE_SCALE = 1.0
CAND_MULTIPLIER = 3

# SDP backend flags: keep deterministic behavior similar to your original runs
os.environ["PYTORCH_SDP_BACKEND"] = "math"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ───────────────────────────── Utils ───────────────────────────────────────
def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_tag(name: str) -> str:
    last = name.split("/")[-1]
    return "".join("_" if c in ".-" else c for c in last).lower()


def quant_tag(q: float) -> str:
    """Turn quantile q into a 'keep percentage' tag, e.g., q=0.9 → '10' (keep 10%)."""
    keep_pct = round((1.0 - q) * 100, 3)
    return ("%g" % keep_pct).replace(".", "_")


def add_identity_mid_hooks(model: HookedTransformer) -> None:
    """Add hook_resid_mid = nn.Identity() per block (naming kept for your viz tooling)."""
    for i, block in enumerate(model.blocks):
        if not hasattr(block, "hook_resid_mid"):
            block.hook_resid_mid = nn.Identity()
            block.hook_resid_mid.name = f"blocks.{i}.hook_resid_mid"


def try_resume_state_dict(model: HookedTransformer, ckpt_path: Path, strict: bool) -> None:
    """Load a state_dict into the TL model; report missing/unexpected keys without failing."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[ckpt] loaded from {ckpt_path}")
    print(f"       missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("       missing keys (head):", missing[:10])
    if unexpected:
        print("       unexpected keys (head):", unexpected[:10])
    pdb.set_trace()


def to_tokens_batched(model: HookedTransformer, texts: List[str]) -> torch.Tensor:
    """Tokenize a batch of texts with left padding to equal length and prepend BOS."""
    return model.to_tokens(texts, prepend_bos=True)  # [B, S]


def make_avg_diff_metric(model: HookedTransformer, true_str: str, false_str: str):
    """
    Average difference on the last position:
    mean over batch of [logit(True_last_tok) - logit(False_last_tok)] at the final token.
    """
    true_tok = model.to_tokens(true_str, prepend_bos=True)[0].tolist()[-1]
    false_tok = model.to_tokens(false_str, prepend_bos=True)[0].tolist()[-1]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_pos = logits.size(1) - 1
        diff = logits[:, last_pos, true_tok] - logits[:, last_pos, false_tok]
        return diff.mean()

    return metric


def extract_clean_corrupt_pairs(examples: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract clean/corrupt pairs from logic_group.prompts; fallback to identical text if malformed."""
    clean, corrupt = [], []
    for ex in examples:
        if isinstance(ex, dict) and ("clean" in ex) and ("corrupt" in ex):
            clean.append(ex["clean"])
            corrupt.append(ex["corrupt"])
        else:
            txt = ex if isinstance(ex, str) else json.dumps(ex, ensure_ascii=False)
            clean.append(txt)
            corrupt.append(txt)
    return clean, corrupt


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_heatmap(scores: torch.Tensor, out_png: Path, title: str = "") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(scores.numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Downstream nodes")
    plt.ylabel("Upstream nodes")
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()


# ───────────────────────────── EAP Core ────────────────────────────────────
def eap_scores_for_subset(
    model: HookedTransformer,
    clean_texts: List[str],
    corrupt_texts: List[str],
    metric_true: str,
    metric_false: str,
    batch_size: int,
) -> "EAPGraph":
    """Run EAP once and return an EAPGraph with eap_scores tensor of shape [U, D] (on CPU)."""
    if EAP is None or EAPGraph is None:
        raise RuntimeError(
            f"Failed to import EAP/EAPGraph from src/. Original error:\n{_IMPORT_ERR}"
        )

    clean_tokens = to_tokens_batched(model, clean_texts)
    corrupt_tokens = to_tokens_batched(model, corrupt_texts)

    metric = make_avg_diff_metric(model, metric_true, metric_false)

    # enable required hooks
    model.set_use_attn_result(True)
    # model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)

    graph = EAP(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupt_tokens,
        metric=metric,
        upstream_nodes=None,       # use EAPGraph defaults
        downstream_nodes=None,     # use EAPGraph defaults
        batch_size=batch_size,
    )
    return graph


def threshold_and_sparse(graph: "EAPGraph", q: float) -> Dict:
    """
    Apply a global quantile threshold to eap_scores and return a sparse dict:
    {
      "shape": [U, D],
      "threshold": float,
      "edges": [ [u_idx, d_idx, score], ... ]   # keep those with score >= threshold
    }
    """
    scores = graph.eap_scores  # Tensor[U, D] (cpu)
    flat = scores.flatten()
    thresh = torch.quantile(flat, q).item() if flat.numel() else 0.0
    keep_mask = scores >= thresh
    u_idx, d_idx = torch.nonzero(keep_mask, as_tuple=True)
    kept_vals = scores[keep_mask]
    edges = torch.stack([u_idx, d_idx, kept_vals], dim=1).tolist()
    return {
        "shape": [int(scores.size(0)), int(scores.size(1))],
        "threshold": float(thresh),
        "edges": edges,
    }


# ──────────────────────── Parsing & Graph Building ─────────────────────────
# relaxed patterns for layer/head parsing
_LAYERS = [re.compile(r"(?:blocks?|layer|blk)[\._\-\/]*(\d+)", re.I)]
_HEADS  = [
    re.compile(r"(?:\.|_|/|^|-)h(?:ead)?[\._\-\/]*(\d+)", re.I),
    re.compile(r"(?:heads?)[\._\-\/]*(\d+)", re.I),
]

def _parse_eap_node_relaxed(node_name: str) -> Tuple[int, str, int]:
    """
    Compatible with:
      - resid_pre.<L>        → (L, 'input', 0)
      - resid_post.<L>       → (L, 'output', 0)
      - mlp.<L>              → (L, 'down_proj', 0)
      - head.<L>.<H>.*       → (L, 'attn_head', H)
    Fallback: layer=0, type='output', index=0.
    """
    name = str(node_name)
    parts = name.split(".")

    if name.startswith("resid_pre."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "input", 0
    if name.startswith("resid_post."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "output", 0
    if name.startswith("mlp."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "down_proj", 0
    if name.startswith("head."):
        L = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        H = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        return L, "attn_head", H

    m = re.search(r"(\d+)", name)
    L = int(m.group(1)) if m else 0
    return L, "output", 0


def build_nx_from_top_edges(
    graph: "EAPGraph",
    n: int = TOP_N_EDGES,
    abs_scores: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
) -> nx.DiGraph:
    """
    Build an nx.DiGraph from the top-n edges of EAPGraph.
    Node key format: "m,L,I,T", original raw name is stored in 'raw'.
    Attention-related edges can be boosted (ATTN_EDGE_SCALE).
    """
    # 1) take more candidates to allow boosting
    raw_n = max(n * CAND_MULTIPLIER, n + 100)
    raw_edges = list(graph.top_edges(n=raw_n, abs_scores=abs_scores))

    boosted_edges = []
    for u_name, v_name, s in raw_edges:
        uL, uT, uI = _parse_eap_node_relaxed(u_name)
        vL, vT, vI = _parse_eap_node_relaxed(v_name)

        # filtering
        if not include_attn and (uT in {"attn_head", "o_proj"} or vT in {"attn_head", "o_proj"}):
            continue
        if not include_mlp and (uT in {"mlp_in", "down_proj"} or vT in {"mlp_in", "down_proj"}):
            continue

        # 2) optional boost for attention edges
        is_attn_edge = (uT in {"attn_head", "o_proj"}) or (vT in {"attn_head", "o_proj"})
        s_boost = np.abs(float(s)) * (1.5 if is_attn_edge else 1.0)
        boosted_edges.append((u_name, v_name, s_boost, uL, uT, uI, vL, vT, vI))

    # 3) sort by boosted magnitude and keep top-n
    boosted_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    boosted_edges = boosted_edges[:n]

    # 4) construct graph
    G = nx.DiGraph()
    def key(L: int, T: str, I: int) -> str:
        return f"m,{L},{I},{T}"

    for (u_name, v_name, s_boost, uL, uT, uI, vL, vT, vI) in boosted_edges:
        if uL < 0 or vL < 0:
            continue
        ukey, vkey = key(uL, uT, uI), key(vL, vT, vI)
        if ukey not in G:
            G.add_node(ukey, raw=str(u_name), layer=int(uL), typ=uT, idx=int(uI))
        if vkey not in G:
            G.add_node(vkey, raw=str(v_name), layer=int(vL), typ=vT, idx=int(vI))
        G.add_edge(ukey, vkey, weight=float(s_boost))

    return G


def _prune_keep_bidir_with_context(G: nx.DiGraph) -> nx.DiGraph:
    """
    Keep nodes that have both incoming and outgoing edges (core nodes),
    plus their immediate predecessors/successors. If empty, return G.
    """
    core = {n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0}
    keep = set(core)
    for n in list(core):
        keep.update(G.predecessors(n))
        keep.update(G.successors(n))
    if not keep:
        return G
    H = G.subgraph(keep).copy()
    return H


# ───────────────────────── Visualization (Graphviz) ────────────────────────
def _node_style_for_typ(typ: str) -> Tuple[str, str]:
    palette = {
        "input":     ("#d7f0d0", "#6ca966"),
        "attn_head": ("#d8e8ff", "#5d7fbf"),
        "o_proj":    ("#d8e8ff", "#5d7fbf"),
        "mlp_in":    ("#fff5cc", "#b8963b"),
        "down_proj": ("#fff5cc", "#b8963b"),
        "output":    ("#ffd7b8", "#c47f42"),
        "mlp":       ("#fff5cc", "#b8963b"),
        "logits":    ("#eadcff", "#6b5fbf"),
    }
    return palette.get(typ, ("#eeeeee", "#888888"))


def _max_layer(nx_graph: nx.DiGraph) -> int:
    maxL = 0
    for n in nx_graph.nodes():
        try:
            maxL = max(maxL, int(nx_graph.nodes[n].get("layer", 0)))
        except Exception:
            pass
    return maxL


def _map_node_id_and_label(attrs: Dict) -> Tuple[str, str, str]:
    L   = int(attrs.get("layer", 0))
    typ = attrs.get("typ", "output")
    idx = int(attrs.get("idx", 0))

    if typ == "input":
        return "INPUT_GLOBAL", "input", "input"

    # merge MLP/residual-like outputs into m{L}
    if typ in {"mlp_in", "down_proj", "output", "o_proj"}:
        return f"M{L}", f"m{L}", "mlp"

    if typ == "attn_head":
        return f"A{L}H{idx}", f"a{L}.h{idx}", "attn_head"

    return f"M{L}", f"m{L}", "mlp"


def _edge_style(weight: float, strong_cut: float) -> Tuple[str, float]:
    """Edge color/width policy: strong→black; positive→blue; negative→red; weak→gray."""
    w = float(abs(weight))
    if w >= strong_cut:
        return ("#000000", 2.2)
    if weight > 0:
        return ("#3b6bd6", 1.3 + 1.2 * w)
    if weight < 0:
        return ("#d64b4b", 1.3 + 1.2 * w)
    return ("#888888", 1.0)


def _collect_input_targets_after_merge(declared_ids: set, edge_dict: Dict) -> List[str]:
    has_incoming = {v for (_, v) in edge_dict.keys()}
    targets = set(declared_ids) - has_incoming - {"INPUT_GLOBAL", "LOGITS"}
    return sorted(targets)


def _collect_logits_sources(nx_graph: nx.DiGraph) -> List[str]:
    """Sources to LOGITS: nodes at the max layer and sink nodes (after merging labels)."""
    Lmax = _max_layer(nx_graph)
    ids = set()
    for n in nx_graph.nodes():
        attrs = nx_graph.nodes[n]
        L = int(attrs.get("layer", 0))
        if L == Lmax:
            nid, _, _ = _map_node_id_and_label(attrs)
            ids.add(nid)
    for n in nx_graph.nodes():
        if nx_graph.out_degree(n) == 0:
            nid, _, _ = _map_node_id_and_label(nx_graph.nodes[n])
            ids.add(nid)
    ids.discard("LOGITS")
    return sorted(ids)


def visualize_circuit_gv(nx_graph: nx.DiGraph, out_file: str) -> None:
    """Graphviz rendering; fallback to a simple NetworkX drawing if graphviz is unavailable."""
    try:
        import graphviz
    except Exception as e:
        print("[viz] graphviz unavailable, falling back to networkx:", e)
        return visualize_circuit_fallback(nx_graph, out_file=out_file)

    nodes = list(nx_graph.nodes())
    ws = [abs(d.get("weight", 0.0)) for _, _, d in nx_graph.edges(data=True)]
    strong_cut = np.quantile(ws, 0.9) if len(ws) >= 4 else (max(ws) if ws else 1.0)

    dot = graphviz.Digraph("circuit", format="png")
    dot.attr(rankdir="LR", splines="spline", concentrate="false",
             nodesep="0.35", ranksep="1.0", bgcolor="white",
             fontname="Helvetica")

    declared: set = set()
    layer_groups: Dict[int, set] = {}
    node_colors: Dict[str, Tuple[str, str]] = {}  # record {node_id: (fill, border)}

    # declare mapped nodes
    for n in nodes:
        attrs = nx_graph.nodes[n]
        node_id, label, color_typ = _map_node_id_and_label(attrs)
        if node_id in declared:
            continue
        fill, border = _node_style_for_typ(color_typ)
        dot.node(node_id, label=label, shape="box", style="rounded,filled",
                 color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
        node_colors[node_id] = (fill, border)
        declared.add(node_id)
        if node_id != "INPUT_GLOBAL":
            L = int(attrs.get("layer", 0))
            layer_groups.setdefault(L, set()).add(node_id)

    # rank=same per layer
    for L, ids in sorted(layer_groups.items()):
        with dot.subgraph(name=f"rank_{L}") as s:
            s.attr(rank="same")
            for nid in sorted(ids):
                s.node(nid)

    # INPUT (leftmost)
    input_id = "INPUT_GLOBAL"
    fill, border = _node_style_for_typ("input")
    dot.node(input_id, label="input", shape="box", style="rounded,filled",
             color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
    node_colors[input_id] = (fill, border)
    with dot.subgraph(name="rank_input") as s:
        s.attr(rank="min")
        s.node(input_id)

    # LOGITS (right)
    logits_id = "LOGITS"
    fill, border = _node_style_for_typ("logits")
    dot.node(logits_id, label="logits", shape="box", style="rounded,filled",
             color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
    node_colors[logits_id] = (fill, border)
    if layer_groups:
        with dot.subgraph(name=f"rank_{max(layer_groups.keys()) + 1}") as s:
            s.attr(rank="same")
            s.node(logits_id)
    
    def _edge_color_from_node(node_id: str) -> str:
        # prefer border color, then fill color, else gray
        fill_col, border_col = node_colors.get(node_id, (None, None))
        if border_col and isinstance(border_col, str) and border_col.startswith("#"):
            return border_col
        if fill_col and isinstance(fill_col, str) and fill_col.startswith("#"):
            return fill_col
        return "#9aa0a6"

    # aggregate multiple edges between same endpoints
    edge_dict: Dict[Tuple[str, str], List[float]] = {}
    for u, v, d in nx_graph.edges(data=True):
        u_id, _, _ = _map_node_id_and_label(nx_graph.nodes[u])
        v_id, _, _ = _map_node_id_and_label(nx_graph.nodes[v])
        if u_id == v_id:
            continue
        w = float(d.get("weight", 0.0))
        edge_dict.setdefault((u_id, v_id), []).append(w)

    # collect outgoing weights per source (for LOGITS edge scaling)
    node_out_weights: Dict[str, List[float]] = {}
    for (u_id, v_id), ws_ in edge_dict.items():
        node_out_weights.setdefault(u_id, []).extend(ws_)

    # draw edges using average weight per pair
    for (u_id, v_id), ws_ in edge_dict.items():
        w = float(np.mean(ws_))
        _, penw = _edge_style(w, strong_cut)
        edge_col = _edge_color_from_node(u_id)
        attrs = {"color": edge_col, "penwidth": f"{penw:.2f}", "arrowsize": "0.7"}
        dot.edge(u_id, v_id, **attrs)

    # INPUT → nodes without any incoming edges (after merging)
    declared_ids_now = set(declared) | {"LOGITS"}
    input_targets = _collect_input_targets_after_merge(declared_ids_now, edge_dict)
    _, inp_border = node_colors.get(input_id, ("#d7f0d0", "#6ca966"))
    for tid in input_targets:
        if tid == "INPUT_GLOBAL":
            continue
        dot.edge(input_id, tid, color=inp_border, penwidth="1.1", arrowsize="0.8")

    # nodes → LOGITS
    for sid in _collect_logits_sources(nx_graph):
        if sid == logits_id:
            continue
        ws_ = node_out_weights.get(sid, [])
        edge_col = _edge_color_from_node(sid)
        if ws_:
            w_signed = float(np.mean(ws_))
            w_abs    = float(np.mean(np.abs(ws_)))
            _, penw = _edge_style(w_signed, strong_cut)
            penw = max(penw, 0.9 + 1.1 * min(w_abs / (strong_cut + 1e-8), 1.0))
            attrs = {"color": edge_col, "penwidth": f"{penw:.2f}", "arrowsize": "0.8"}
        else:
            attrs = {"color": edge_col, "penwidth": "0.9", "arrowsize": "0.8"}
        dot.edge(sid, logits_id, **attrs)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(filename=out_path.with_suffix("").as_posix(), cleanup=True)
    dot.render(filename=out_path.with_suffix(".pdf").as_posix(), format="pdf", cleanup=True)
    print(f"[viz] saved: {out_file}")


def visualize_circuit_fallback(nx_graph: nx.DiGraph, out_file: str) -> None:
    """Simple NetworkX fallback when graphviz is not installed."""
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(nx_graph, seed=0)
    plt.figure(figsize=(10, 6))
    nx.draw(nx_graph, pos, with_labels=False, node_size=300, width=0.8, arrows=True)
    plt.tight_layout()
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()
    print(f"[viz-fallback] saved: {out_file}")


def visualize_top_edges_as_circuit(
    graph: "EAPGraph",
    out_file: str,
    n: int = TOP_N_EDGES,
    abs_scores: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
) -> None:
    """EAPGraph → NetworkX → Graphviz PNG."""
    # debug: quick parse preview
    for u, v, s in graph.top_edges(n=min(8, n), abs_scores=True):
        uL, uT, uI = _parse_eap_node_relaxed(u)
        vL, vT, vI = _parse_eap_node_relaxed(v)
        print(f"[parsed] U=({uL},{uT},{uI})  V=({vL},{vT},{vI})  |S|={abs(s):.3f}   rawU={u}  rawV={v}")

    G = build_nx_from_top_edges(
        graph, n=n, abs_scores=abs_scores,
        include_attn=include_attn, include_mlp=include_mlp,
    )
    G = _prune_keep_bidir_with_context(G)
    visualize_circuit_gv(G, out_file=out_file)


# ───────────────────────────── CLI & Main ──────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="TransformerLens-compatible model name (e.g., gpt2-small or TL-ported name)")
    p.add_argument("--data_file", default=DEFAULT_DATA,
                   help="JSON with logic_groups[*].prompts[*].{clean,corrupt,...}")
    p.add_argument("--quants", nargs="+", type=float, default=[0.5],
                   help="Quantile thresholds; keep scores >= q (i.e., keep (1-q) proportion)")
    p.add_argument("--subset_k", type=int, default=10,
                   help="Per split, sample size k for partA and partB each")
    p.add_argument("--splits", type=int, default=1, help="Number of random splits")
    p.add_argument("--seed", type=int, default=0, help="Random seed base")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for EAP forward/backward")
    p.add_argument("--out_root", default=DEFAULT_OUT, help="Output root directory")
    p.add_argument("--resume", type=Path, help="Optional: path to state_dict to load into TL model")
    p.add_argument("--strict_resume", action="store_true", help="Strict key matching when loading state_dict")
    p.add_argument("--ans_true", default=" True", help="Positive token for avg_diff (default ' True')")
    p.add_argument("--ans_false", default=" False", help="Negative token for avg_diff (default ' False')")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seeds
    set_seed_all(args.seed)

    # 1) load TL model
    print(f"[load] HookedTransformer.from_pretrained({args.model}) on {device}")
    model = HookedTransformer.from_pretrained(args.model, device=device)

    # some configs may not exist; set defensively
    try:
        model.cfg.parallel_attn_mlp = False
    except Exception:
        pass
    add_identity_mid_hooks(model)

    # 2) resume if provided
    run_tag = Path(args.resume).stem if args.resume else "origin"
    if args.resume is not None:
        try_resume_state_dict(model, args.resume, args.strict_resume)
    print(f"[info] Using run tag: {run_tag}")

    # 3) directories
    out_root = Path(args.out_root).resolve()
    dataset_tag = Path(args.data_file).stem
    model_dir = out_root / dataset_tag / model_tag(args.model)
    for q in args.quants:
        (model_dir / quant_tag(q) / run_tag / "figures").mkdir(parents=True, exist_ok=True)

    # 4) load data
    with open(args.data_file, "r", encoding="utf-8") as f:
        logic_groups = json.load(f)

    # 5) iterate logic groups
    for idx, logic in enumerate(logic_groups):
        examples = logic.get("prompts", [])
        if not examples:
            continue

        logic_name = f"logic_{idx:03d}"
        if idx < 8:
            continue
        print(f"\n▶ {logic_name} — {len(examples)} prompts")

        # 6) sampling/splitting
        k = min(args.subset_k, max(len(examples) // 2, 1))
        if dataset_tag == "level_3":
            k = min(k, 4)

        for split in range(1, args.splits + 1):
            seed_i = args.seed + split - 1
            rng = random.Random(seed_i)
            shuffled = examples[:]
            # rng.shuffle(shuffled)
            partA, partB = shuffled[:k], shuffled[k: 2 * k]

            # 6.1) subset A
            clean_A, corrupt_A = extract_clean_corrupt_pairs(partA)
            graph_A = eap_scores_for_subset(
                model, clean_A, corrupt_A, args.ans_true, args.ans_false, args.batch_size
            )
            for q in args.quants:
                qtag = quant_tag(q)
                subdir = model_dir / qtag / run_tag
                json_path = subdir / f"{logic_name}_split{seed_i}_partA.json"
                save_json(threshold_and_sparse(graph_A, q), json_path)

                viz_dir = subdir / "figures"
                viz_dir.mkdir(parents=True, exist_ok=True)
                visualize_top_edges_as_circuit(
                    graph_A,
                    out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partA_circuit_top100.png"),
                    n=TOP_N_EDGES,
                    abs_scores=True,
                    include_attn=True,
                    include_mlp=True,
                )
                heat_path = viz_dir / f"{logic_name}_split{seed_i}_partA.png"
                save_heatmap(graph_A.eap_scores, heat_path, title=f"{logic_name} partA (q={q:.3f})")
                print(f"   ✓ partA  q={q:.3f} → {json_path.relative_to(model_dir)}")

            # 6.2) subset B (if exists)
            if partB:
                clean_B, corrupt_B = extract_clean_corrupt_pairs(partB)
                graph_B = eap_scores_for_subset(
                    model, clean_B, corrupt_B, args.ans_true, args.ans_false, args.batch_size
                )
                for q in args.quants:
                    qtag = quant_tag(q)
                    subdir = model_dir / qtag / run_tag
                    json_path = subdir / f"{logic_name}_split{seed_i}_partB.json"
                    save_json(threshold_and_sparse(graph_B, q), json_path)

                    viz_dir = subdir / "figures"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    visualize_top_edges_as_circuit(
                        graph_B,
                        out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partB_circuit_top100.png"),
                        n=TOP_N_EDGES,
                        abs_scores=True,
                        include_attn=True,
                        include_mlp=True,
                    )
                    heat_path = viz_dir / f"{logic_name}_split{seed_i}_partB.png"
                    save_heatmap(graph_B.eap_scores, heat_path, title=f"{logic_name} partB (q={q:.3f})")
                    print(f"   ✓ partB  q={q:.3f} → {json_path.relative_to(model_dir)}")

    print("\n✅ Finished for all TOP_QUANTS.")


if __name__ == "__main__":
    main()