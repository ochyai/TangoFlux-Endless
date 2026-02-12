#!/usr/bin/env python3
"""TangoFlux MPS vs CPU Benchmark"""

import time
import torch
import torchaudio
from tangoflux import TangoFluxInference

PROMPT = "A gentle rain falling on leaves with distant thunder"
DURATION = 10
STEPS = 50

def benchmark(device_name):
    print(f"\n{'='*50}")
    print(f"  Device: {device_name.upper()}")
    print(f"{'='*50}")

    # Load model
    t0 = time.time()
    model = TangoFluxInference(name="declare-lab/TangoFlux", device=device_name)
    load_time = time.time() - t0
    print(f"  Model load:  {load_time:.2f}s")

    # Warmup (1 step)
    print("  Warmup...")
    with torch.no_grad():
        _ = model.generate(PROMPT, steps=1, duration=1)
    if device_name == "mps":
        torch.mps.synchronize()

    # Generate
    print(f"  Generating {DURATION}s audio, {STEPS} steps...")
    t1 = time.time()
    with torch.no_grad():
        audio = model.generate(PROMPT, steps=STEPS, duration=DURATION)
    if device_name == "mps":
        torch.mps.synchronize()
    gen_time = time.time() - t1

    torchaudio.save(f"benchmark_{device_name}.wav", audio, 44100)

    rtf = gen_time / DURATION
    print(f"  Generation:  {gen_time:.2f}s")
    print(f"  RTF:         {rtf:.3f}x (< 1.0 = faster than realtime)")
    print(f"  Audio shape: {audio.shape}")

    # Cleanup
    del model
    if device_name == "mps":
        torch.mps.empty_cache()

    return {"device": device_name, "load": load_time, "gen": gen_time, "rtf": rtf}


if __name__ == "__main__":
    print("TangoFlux Benchmark")
    print(f"Prompt: '{PROMPT}'")
    print(f"Duration: {DURATION}s | Steps: {STEPS}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    results = []

    # MPS first
    if torch.backends.mps.is_available():
        results.append(benchmark("mps"))

    # CPU
    results.append(benchmark("cpu"))

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Device':<8} {'Load':>8} {'Generate':>10} {'RTF':>8}")
    print(f"  {'-'*36}")
    for r in results:
        print(f"  {r['device'].upper():<8} {r['load']:>7.2f}s {r['gen']:>9.2f}s {r['rtf']:>7.3f}x")

    if len(results) == 2:
        speedup = results[1]["gen"] / results[0]["gen"]
        print(f"\n  MPS speedup: {speedup:.2f}x faster than CPU")
