"""Held-out real workload: LLM serving client (Ollama HTTP API).

Sends generation requests to a local Ollama server with Poisson inter-arrival
times (``--rate`` requests/s), each asking for ``--tokens`` new tokens, so the GPU
alternates between busy decoding and idle gaps as an interactive service does.
Requests run concurrently in threads; the server queues them.

``--warmup`` sends one request (loads the model onto the GPU) and exits.
Otherwise prints ``READY`` and runs until terminated.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Explain how a Kubernetes scheduler chooses a node for a pod.",
    "Summarise the trade-offs between energy efficiency and latency in data centres.",
    "Write a short Python function that merges two sorted lists.",
    "Describe the water cycle to a ten-year-old.",
    "List five practical tips for reducing the carbon footprint of software.",
    "What is the difference between a process and a thread?",
    "Draft a polite e-mail asking a colleague to review a paper draft.",
    "Explain gradient descent in three sentences.",
]


def generate(url: str, model: str, prompt: str, tokens: int, timeout: float = 600.0) -> float:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"num_predict": tokens, "temperature": 0.7}}).encode()
    req = urllib.request.Request(f"{url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()

    if args.warmup:
        generate(args.url, args.model, "Say hello.", 8)
        return 0

    rng = random.Random(args.seed)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    done = {"n": 0, "lat": 0.0}
    lock = threading.Lock()

    def _one(prompt: str) -> None:
        try:
            lat = generate(args.url, args.model, prompt, args.tokens)
            with lock:
                done["n"] += 1
                done["lat"] += lat
        except Exception as e:  # keep serving; report on stderr
            print(f"request failed: {e!r}", file=sys.stderr, flush=True)

    print("READY", flush=True)
    while not stop.is_set():
        threading.Thread(target=_one, args=(rng.choice(PROMPTS),), daemon=True).start()
        stop.wait(rng.expovariate(args.rate))
    with lock:
        n = done["n"]
        print(f"completed {n} requests, mean latency {done['lat'] / max(1, n):.2f}s",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
