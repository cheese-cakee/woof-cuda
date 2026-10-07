"""Run a resumable matched suite, saving every completed configuration."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

from transformers import AutoTokenizer

import bench_suite


def stop(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("configs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-names", help="Comma-separated suite prompt names")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    os.environ["WOOF_SUITE_LOGDIR"] = str(args.out.parent / "server-logs")
    lab = Path(os.environ.setdefault("WOOF_LAB", str(Path.home() / "woof-lab")))
    configs = json.loads(args.configs.read_text())
    tokenizer = AutoTokenizer.from_pretrained(lab / "models" / f"{args.model}-mlx")
    prompts = [{"name": p["name"], "kind": p["kind"],
                "ids": tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True,
                                                     enable_thinking=False, return_dict=False)}
               for p in json.loads((Path(__file__).resolve().parent / "prompts/husky_suite.json").read_text())]
    if args.prompt_names:
        names_requested = args.prompt_names.split(",")
        available = {p["name"] for p in prompts}
        if any(name not in available for name in names_requested):
            raise ValueError("Unknown suite prompt name")
        prompts = [p for p in prompts if p["name"] in names_requested]
    metadata = {"model": args.model, "configs": configs, "max_tokens": args.max_tokens,
                "runs_requested": args.runs,
                "prompts": [{"name": p["name"], "kind": p["kind"], "prompt_tokens": len(p["ids"])} for p in prompts]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        report = json.loads(args.out.read_text())
        if any(report[key] != value for key, value in metadata.items()):
            raise ValueError("Saved run has different inputs; use a new output path")
    else:
        report = {**metadata, "runs": {name: [] for name in configs}, "batches": []}
    server_bin = str(lab / "llama.cpp/build-cuda/bin/llama-server")
    gguf = str(lab / "gguf" / f"{args.model}-Q4_1-exact.gguf")
    names = list(configs)
    for run in range(args.runs):
        for name in names if run % 2 == 0 else names[::-1]:
            if len(report["runs"][name]) > run:
                continue
            print(f"run {run + 1}/{args.runs} config {name}", flush=True)
            started = time.time()
            environment = configs[name].get("env", {}) if isinstance(configs[name], dict) else {}
            previous = {key: os.environ.get(key) for key in environment}
            os.environ.update(environment)
            try:
                result = bench_suite.run_config(server_bin, gguf, configs[name], prompts, args.max_tokens)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            report["runs"][name].append(result)
            report["batches"].append({"run": run, "config": name, "started_unix": started,
                                       "finished_unix": time.time()})
            temporary = args.out.with_suffix(".tmp")
            temporary.write_text(json.dumps(report))
            temporary.replace(args.out)
    bench_suite.print_summary(names, prompts, report["runs"])
    reference = names[0]
    for name in names:
        differences = []
        for run, samples in enumerate(report["runs"][name]):
            for i, sample in enumerate(samples):
                ref = report["runs"][reference][run][i]["tokens"]
                candidate = sample["tokens"]
                if candidate != ref:
                    position = next((j for j, (a, b) in enumerate(zip(ref, candidate)) if a != b),
                                    min(len(ref), len(candidate)))
                    differences.append({"run": run, "prompt": prompts[i]["name"], "position": position,
                                        "reference": ref[position:position + 1], "candidate": candidate[position:position + 1]})
        print(f"{name}: {len(differences)}/{args.runs * len(prompts)} outputs differ from {reference}", flush=True)
        report.setdefault("differences", {})[name] = differences
    args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
