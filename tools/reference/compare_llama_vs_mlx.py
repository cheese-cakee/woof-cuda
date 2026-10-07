"""Greedy-decode MLX's exact prompt tokens with llama-server and compare token by token against the MLX reference."""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8091


def post(path, payload):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def wait_until_healthy(server):
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError("llama-server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_bin")
    parser.add_argument("gguf")
    parser.add_argument("mlx_json", type=Path)
    parser.add_argument("--save", type=Path, help="write llama.cpp tokens here")
    parser.add_argument("--expect", type=Path, help="compare against tokens saved by --save instead of MLX")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args, extra = parser.parse_args(argv[:split]), argv[split + 1:]  # flags after -- go to llama-server

    reference = json.loads(args.mlx_json.read_text())["results"]
    expected_override = json.loads(args.expect.read_text()) if args.expect else None
    server = subprocess.Popen([args.server_bin, "-m", args.gguf, "-ngl", "99", "-fa", "on", "-c", "8192",
                               "--port", str(PORT), "-np", "1", *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    saved = []
    try:
        wait_until_healthy(server)
        for index, ref in enumerate(reference):
            if "error" in ref:
                print(f"prompt {ref['prompt_tokens']:5d} tok | MLX failed ({ref['error'][:40]}...), llama.cpp not compared")
                saved.append(None)
                continue
            expected = expected_override[index] if expected_override else ref["generated_ids"]
            result = post("/completion", {"prompt": ref["prompt_ids"], "n_predict": len(expected), "temperature": 0,
                                          "top_k": 1, "cache_prompt": False, "return_tokens": True})
            got = result["tokens"][: len(expected)]
            saved.append(got)
            timings = result["timings"]
            diverge = first_divergence(expected, got)
            if diverge is None and len(got) == len(expected):
                verdict = f"identical {len(expected)}/{len(expected)} tokens"
            elif diverge is None:
                verdict = f"identical prefix, llama.cpp stopped at {len(got)}/{len(expected)}"
            elif expected_override:
                verdict = f"diverge at token {diverge}/{len(expected)}"
            else:
                verdict = f"diverge at token {diverge}/{len(expected)}, MLX top1-top2 margin there {ref['top2_margins'][diverge]}"
            accepted = ""
            if timings.get("draft_n"):
                accepted = f" | draft accepted {timings['draft_n_accepted']}/{timings['draft_n']}"
            print(f"prompt {ref['prompt_tokens']:5d} tok | {verdict} | llama.cpp prefill "
                  f"{timings['prompt_per_second']:.0f} tok/s decode {timings['predicted_per_second']:.1f} tok/s{accepted} | "
                  f"MLX prefill {ref['prefill_tok_s']} decode {ref['decode_tok_s']}")
    finally:
        server.terminate()
        server.wait()
    if args.save:
        args.save.write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
