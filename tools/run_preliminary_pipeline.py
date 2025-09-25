#!/usr/bin/env python3
from __future__ import annotations
import subprocess, itertools, argparse, os
from pathlib import Path
from glob import glob
import pdb  
from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------- helpers -----------------------------
DISTANCE_MODULES_DEFAULT = {
    "pot": "src.preliminary.distance.pot_distance",
    "edit": "src.preliminary.distance.edit_distance",
    "jaccard": "src.preliminary.distance.jaccard_distance",
}

REQ_PNGS_DEFAULT = ["scatter.png", "scatter_binned.png",
                    "scatter_sliding.png", "scatter_acc.png"]

def run(cmd, **kw):
    print("➤", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_distance_mapping(kv_list: list[str]) -> dict[str, str]:
    mapping = DISTANCE_MODULES_DEFAULT.copy()
    for kv in kv_list or []:
        if "=" not in kv:
            raise ValueError(f"--distance-map expects key=module, got: {kv}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Bad mapping: {kv}")
        mapping[k] = v
    return mapping

def slug_lr(lr: float | str) -> str:
    s = str(lr)
    return s.replace(".", "p").replace("-", "m")

# ----------------------------- main -----------------------------
def main(args: argparse.Namespace):
    distance_map = parse_distance_mapping(args.distance_map)
    if args.distances is None or len(args.distances) == 0:
        use_distances = list(distance_map.keys())
    else:
        unknown = [d for d in args.distances if d not in distance_map]
        if unknown:
            raise ValueError(f"Unknown distances: {unknown}. "
                             f"Known: {sorted(distance_map.keys())}")
        use_distances = args.distances

    resume = args.resume if (args.resume and str(args.resume).strip()) else None
    resume_tag = Path(resume).stem if resume is not None else "origin"
    dataset_tag = Path(args.src_json).stem
    data_file = DATA_DIR / "corrupt" / f"{dataset_tag}.json"

    if args.run_circuits:
        for seed in args.seeds:
            pattern = str(PROJECT_ROOT / f"results/output/attr_scores/{dataset_tag}/{args.model_id}/10/{resume_tag}/seed{seed}_split*.json")
            matching_dirs = glob(pattern)
            if args.force or not matching_dirs:
                cmd = ["python", "-m", "src.preliminary.circuit.circuit_aio",
                       "--seed", str(seed),
                       "--data_file", data_file,
                       "--out_root", str(PROJECT_ROOT / "results/output/attr_scores")]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    if args.run_distances:
        for seed in args.seeds:
            for dist_name in use_distances:
                module = distance_map[dist_name]
                csv_path = PROJECT_ROOT / f"results/output/distance/{dataset_tag}/{dist_name}/{resume_tag}/seed{seed}.csv"
                if args.force or not csv_path.exists():
                    input_dir = (PROJECT_ROOT /
                                 f"results/output/attr_scores/{dataset_tag}/{args.model_id}/10/{resume_tag}")
                    ensure_dir(csv_path.parent)
                    cmd = ["python", "-m", module,
                           "--seed", str(seed),
                           "--input", str(input_dir),
                           "--out_csv", str(csv_path)]
                    if resume is not None:
                        cmd += ["--resume", str(resume)]
                    run(cmd)

    if args.run_perlogic:
        for lr in args.lrs:
            lr_slug = slug_lr(lr)
            csv_root = PROJECT_ROOT / f"results/output/perlogic/{dataset_tag}/{lr_slug}/{resume_tag}"
            if args.force or not csv_root.exists():
                ensure_dir(csv_root)
                cmd = ["python", "-m", "src.preliminary.edit.perlogic_delta_batch",
                       "--lr", str(lr),
                       "--src_json", args.src_json,
                       "--out_root", str(PROJECT_ROOT / "results/output/perlogic")]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    if args.run_lora_edit:
        for lr in args.lrs:
            cmd  = [
                "python", "-m", "src.lora_edit",
                "--lr", str(lr),
                "--src_json", args.src_json,
            ]
            if resume is not None:
                cmd += ["--resume", str(resume)]
            run(cmd)
            

    if args.plot_mode == None:
        return
    if args.plot_mode in {"combined", "both"}:
        for dist_name in use_distances:
            for lr in args.lrs:
                seeds_str = ",".join(str(s) for s in args.seeds)
                cmd = [
                    "python", "-m", "tools.plot_preliminary_combined_int",
                    "--distance", dist_name,
                    "--seeds", seeds_str,
                    "--lrs", str(lr),
                    "--dataset_tag", str(dataset_tag),
                    "--combine", str(args.combine_size or 0),
                ]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    if args.plot_mode in {"single", "both"}:
        req_pngs = args.req_pngs or REQ_PNGS_DEFAULT
        for seed, dist_name, lr in itertools.product(args.seeds, use_distances, args.lrs):
            lr_slug = slug_lr(lr)
            out_dir = PROJECT_ROOT / f"results/figures/{dist_name}/seed{seed}/lr{lr_slug}/{resume_tag}"
            need = args.force or not out_dir.exists() or not all((out_dir / f).exists() for f in req_pngs)
            if need:
                ensure_dir(out_dir)
                cmd = [
                    "python", "-m", "tools.plot_preliminary_intra_inter",
                    "--distance", dist_name,
                    "--seed", str(seed),
                    "--dataset_tag", str(dataset_tag),
                    "--lr", str(lr),
                ]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd, cwd=out_dir)
        
    if args.plot_mode in {"cluster"}:
        for dist_name in use_distances:
            cmd = [
                "python", "-m", "tools.plot_cluster",
                "--distance", dist_name,
                "--lrs", " ".join(str(lr) for lr in args.lrs),
                "--seeds", " ".join(str(s) for s in args.seeds),
                "--dataset_tag", str(dataset_tag),
                "--combine", str(args.combine_size or 0),
            ]
            if resume is not None:
                cmd += ["--subdir", str(resume_tag)]
            run(cmd)

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Batch run seed × distance × lr; all use `python -m pkg.module …` calls (parametric version)"
    )
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0,1,2,3,4,5,6,7,8,9,10],
                    help="Random seeds")
    ap.add_argument("--model-id", default="qwen2_5_3b_instruct",
                    help="Model ID")
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5],
                    help="Learning rates")
    ap.add_argument("--distances", nargs="*", default=None,
                    help=f"Which distance metrics to use (default all): {','.join(DISTANCE_MODULES_DEFAULT.keys())}")
    ap.add_argument("--distance-map", nargs="*", default=None,
                    help=("Override/add distance name to module mapping, e.g.: "
                          "pot=src.preliminary.distance.pot_distance "
                          "edit=src.preliminary.distance.edit_distance"))
    ap.add_argument("--req-pngs", nargs="*", default=REQ_PNGS_DEFAULT,
                    help="Required PNGs for single plotting mode")

    ap.add_argument("--plot-mode", choices=["single", "combined", "both", "cluster"],
                    default=None,
                    help="Plotting mode")
    ap.add_argument("--combine-size", type=int, default=10,
                    help="Number of runs to combine in combined mode; 0 means all")

    ap.add_argument("--resume", default=None,
                    help="Optional checkpoint path; empty means origin")
    ap.add_argument("--force", action="store_true",
                    help="Force rerun/redraw")

    def str2bool(v: str) -> bool:
        return v.lower() in {'true', '1', 'yes'}

    ap.add_argument("--run-circuits", default=False, type=str2bool,
                    help="Run circuits step")
    ap.add_argument("--run-distances", default=False, type=str2bool,
                    help="Run distances step")
    ap.add_argument("--run-perlogic", default=False, type=str2bool,
                    help="Run per-logic Δ-accuracy step")
    ap.add_argument("--run-lora-edit", default=False, type=str2bool,
                    help="Run LoRA edit step")
    ap.add_argument("--src_json", default=DATA_DIR / "logic/level_1.json")
    args = ap.parse_args()

    main(args)