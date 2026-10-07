"""Husky-style benchmark of llama-server configurations on the prompt suite.

Canary run first, then N timed runs with configuration order alternated. Reports per-prompt median and range of
decode tok/s, prefill tok/s and draft acceptance, and checks every configuration's tokens against the first one.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

PORT = 8092


def post(path, payload):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def start_server(server_bin, gguf, flags):
    log_dir = os.environ.get("WOOF_SUITE_LOGDIR")
    log = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log = (Path(log_dir) / f"server-{time.time_ns()}.log").open("w")
    try:
        server = subprocess.Popen([server_bin, "-m", gguf, "-ngl", "99", "-fa", "on", "-c", "16384", "-np", "1",
                                   "--host", "127.0.0.1", "--port", str(PORT), *flags],
                                  stdout=subprocess.DEVNULL, stderr=log or subprocess.DEVNULL)
    finally:
        if log:
            log.close()
    for _ in range(180):
        if server.poll() is not None:
            raise RuntimeError(f"llama-server exited for flags {flags}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    return server
        except OSError:
            time.sleep(1)
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    raise RuntimeError("llama-server did not become healthy")


def generate(prompt_ids, max_tokens):
    started = time.perf_counter()
    result = post("/completion", {"prompt": prompt_ids, "n_predict": max_tokens, "temperature": 0, "top_k": 1,
                                  "cache_prompt": False, "return_tokens": True})
    seconds = time.perf_counter() - started
    t = result["timings"]
    return {"tokens": result["tokens"], "decode_tok_s": t["predicted_per_second"], "prefill_tok_s": t["prompt_per_second"],
            "draft": t.get("draft_n", 0), "accepted": t.get("draft_n_accepted", 0), "timings": t, "wall_seconds": seconds}


def run_config(server_bin, gguf, config, prompts, max_tokens):
    if isinstance(config, dict):  # {"bin": path, "flags": [...]} overrides the server binary for this config
        server_bin = os.path.expandvars(config["bin"])
        flags = [os.path.expandvars(flag) for flag in config["flags"]]
        gguf = os.path.expandvars(config.get("gguf", gguf))
    else:
        flags = config
    server = start_server(server_bin, gguf, flags)
    try:
        generate(prompts[0]["ids"], 32)  # canary
        return [generate(p["ids"], max_tokens) for p in prompts]
    finally:
        server.terminate()
        server.wait()


def flags_of(config):
    return config["flags"] if isinstance(config, dict) else config


def median_range(values):
    return f"{statistics.median(values):7.1f} [{min(values):.1f}-{max(values):.1f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_bin")
    parser.add_argument("gguf")
    parser.add_argument("tokenizer_dir")
    parser.add_argument("suite", type=Path)
    parser.add_argument("configs", type=Path, help='JSON {"name": [llama-server flags] or {"bin": path, "flags": [...]}, ...}; first is the reference')
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    prompts = [{"name": p["name"], "kind": p["kind"],
                "ids": tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, enable_thinking=False, return_dict=False)}
               for p in json.loads(args.suite.read_text())]
    configs = json.loads(args.configs.read_text())
    names = list(configs)

    runs = {name: [] for name in names}
    for run in range(args.runs):
        order = names if run % 2 == 0 else names[::-1]
        for name in order:
            print(f"run {run + 1}/{args.runs} config {name}", flush=True)
            runs[name].append(run_config(args.server_bin, args.gguf, configs[name], prompts, args.max_tokens))

    reference = runs[names[0]][0]
    report = {"prompts": [{"name": p["name"], "kind": p["kind"], "prompt_tokens": len(p["ids"])} for p in prompts],
              "configs": configs, "runs": runs}
    args.out.write_text(json.dumps(report))

    print_summary(names, prompts, runs)
    for name in names:
        print(f"\n== {name}: {' '.join(flags_of(configs[name])) or '(plain)'}")
        print(f"{'prompt':24s} {'in':>5s} {'out':>4s} {'decode tok/s median [range]':>30s} {'prefill':>8s} {'accept':>8s}  tokens vs {names[0]}")
        for i, p in enumerate(prompts):
            samples = [r[i] for r in runs[name]]
            out_tokens = samples[0]["tokens"]
            same_runs = all(s["tokens"] == out_tokens for s in samples)
            same_ref = out_tokens == reference[i]["tokens"]
            accept = f"{samples[0]['accepted']}/{samples[0]['draft']}" if samples[0]["draft"] else "-"
            verdict = ("identical" if same_ref else "DIFFERS") + ("" if same_runs else ", unstable across runs")
            print(f"{p['name']:24s} {len(p['ids']):5d} {len(out_tokens):4d} {median_range([s['decode_tok_s'] for s in samples]):>30s} "
                  f"{statistics.median(s['prefill_tok_s'] for s in samples):8.0f} {accept:>8s}  {verdict}")


def print_summary(names, prompts, runs):
    def median_speed(name, i):
        return statistics.median(r[i]["decode_tok_s"] for r in runs[name])

    def decode_seconds(run):
        return sum(len(s["tokens"]) / s["decode_tok_s"] for s in run)

    kinds = sorted({p["kind"] for p in prompts})
    print("\n== summary: geometric-mean decode speedup vs " + names[0])
    print(f"{'config':28s} " + " ".join(f"{k:>9s}" for k in kinds) + f" {'all':>7s} {'decode s':>9s}")
    for name in names:
        speedups = {i: median_speed(name, i) / median_speed(names[0], i) for i in range(len(prompts))}
        cells = []
        for group in kinds + [None]:
            values = [v for i, v in speedups.items() if group is None or prompts[i]["kind"] == group]
            cells.append(math.exp(sum(map(math.log, values)) / len(values)))
        seconds = statistics.median(decode_seconds(run) for run in runs[name])
        print(f"{name:28s} " + " ".join(f"{c:9.2f}" for c in cells[:-1]) + f" {cells[-1]:7.2f} {seconds:9.1f}")


if __name__ == "__main__":
    main()
