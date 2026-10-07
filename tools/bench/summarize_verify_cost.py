"""Print c(n) = time(n tokens) / time(1 token) from llama-bench JSON."""
import json
import sys

rows = json.load(open(sys.argv[1]))
base = next(r["avg_ns"] for r in rows if r["n_prompt"] == 1)
for r in rows:
    print(f"n={r['n_prompt']:3d}  {r['avg_ns'] / 1e6:6.2f} ms  c(n)={r['avg_ns'] / base:5.2f}  sd={100 * r['stddev_ns'] / r['avg_ns']:4.1f}%")
