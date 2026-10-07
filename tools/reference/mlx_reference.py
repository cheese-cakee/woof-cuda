"""Greedy reference run of an MLX checkpoint: token ids, top-2 logit margins and timings, written as JSON."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def greedy(model, prompt_ids, max_new_tokens, eos_ids):
    cache = make_prompt_cache(model)
    tokens = mx.array(prompt_ids)[None]
    start = time.perf_counter()
    logits = model(tokens, cache=cache)[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    prefill_s = time.perf_counter() - start

    generated, margins = [], []
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        top2 = mx.topk(logits[0], 2)
        next_id = mx.argmax(logits[0]).item()
        margins.append(round(abs((top2[1] - top2[0]).item()), 4))
        generated.append(next_id)
        if next_id in eos_ids:
            break
        logits = model(mx.array([[next_id]]), cache=cache)[:, -1, :].astype(mx.float32)
        mx.eval(logits)
    decode_s = time.perf_counter() - start
    return generated, margins, prefill_s, decode_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompts", type=Path, help="JSON list of chat message lists")
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-think", action="store_true", help="render the chat template with enable_thinking=False")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    model, tokenizer = load(str(args.model_dir))
    eos_ids = set(tokenizer.eos_token_ids)

    results = []
    for messages in json.loads(args.prompts.read_text()):
        template_kwargs = {"enable_thinking": False} if args.no_think else {}
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **template_kwargs)
        mx.clear_cache()
        try:
            greedy(model, prompt_ids, 8, eos_ids)  # warmup
            generated, margins, prefill_s, decode_s = greedy(model, prompt_ids, args.max_new_tokens, eos_ids)
        except RuntimeError as error:
            results.append({"prompt_tokens": len(prompt_ids), "prompt_ids": prompt_ids, "error": str(error)})
            print(f"prompt {len(prompt_ids):5d} tok | FAILED: {error}")
            continue
        results.append({
            "prompt_tokens": len(prompt_ids),
            "prompt_ids": prompt_ids,
            "generated_ids": generated,
            "top2_margins": margins,
            "text": tokenizer.decode(generated),
            "prefill_tok_s": round(len(prompt_ids) / prefill_s, 1),
            "decode_tok_s": round((len(generated) - 1) / decode_s, 1) if len(generated) > 1 else None,
        })
        print(f"prompt {len(prompt_ids):5d} tok | prefill {results[-1]['prefill_tok_s']:8.1f} tok/s | "
              f"decode {results[-1]['decode_tok_s']} tok/s | {results[-1]['text'][:60]!r}")
    args.out.write_text(json.dumps({"device": args.device, "model": str(args.model_dir), "results": results}, indent=1))


if __name__ == "__main__":
    main()
