#!/usr/bin/env python3
"""
Comprehensive benchmark for arXiv paper.
Measures: latency distribution, RTF, spectral quality, numerical precision.
Outputs JSON results for LaTeX table generation.

Usage:
    cd /Users/yoichiochiai/計算機の自然
    source venv/bin/activate
    python /Users/yoichiochiai/TangoFlux-Endless/benchmarks/paper_benchmark.py
"""

import sys
import os
import time
import json
import gc
import numpy as np

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(os.path.expanduser("~"), "計算機の自然")
COREML_DIR = os.path.join(PROJECT_DIR, "coreml_models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTS = [
    "A gentle rain falling on leaves with distant thunder",
    "Ocean waves crashing on a rocky shore with seagulls",
    "A busy cafe with people talking and coffee machines",
    "Wind blowing through a dense forest with birdsong",
    "A crackling campfire in a quiet mountain clearing",
]

DURATION = 18
SAMPLE_RATE = 44100


def spectral_analysis(audio_np, sr=SAMPLE_RATE):
    """Compute spectral features of generated audio."""
    from scipy import signal as sig

    results = {}

    # Basic stats
    results["peak_amplitude"] = float(np.abs(audio_np).max())
    results["rms"] = float(np.sqrt(np.mean(audio_np ** 2)))
    results["dynamic_range_db"] = float(
        20 * np.log10(results["peak_amplitude"] / (results["rms"] + 1e-10))
    )

    # Spectral centroid (brightness)
    n_fft = 2048
    hop = 512
    f, t_spec, Sxx = sig.spectrogram(audio_np, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    Sxx_mag = np.abs(Sxx)
    spectral_centroid = np.sum(f[:, None] * Sxx_mag, axis=0) / (np.sum(Sxx_mag, axis=0) + 1e-10)
    results["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    results["spectral_centroid_std"] = float(np.std(spectral_centroid))

    # Spectral rolloff (95% energy)
    cumsum = np.cumsum(Sxx_mag, axis=0)
    total = cumsum[-1, :]
    rolloff_idx = np.argmax(cumsum >= 0.95 * total[None, :], axis=0)
    rolloff_freq = f[rolloff_idx]
    results["spectral_rolloff_mean"] = float(np.mean(rolloff_freq))

    # Zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio_np))) > 0)
    results["zero_crossing_rate"] = float(zcr)

    # Silence ratio (frames below -60dB)
    frame_energy = np.array([
        np.sqrt(np.mean(audio_np[i:i + hop] ** 2))
        for i in range(0, len(audio_np) - hop, hop)
    ])
    silence_threshold = 10 ** (-60 / 20)
    results["silence_ratio"] = float(np.mean(frame_energy < silence_threshold))

    # Spectral flatness (tonality vs noise)
    geo_mean = np.exp(np.mean(np.log(Sxx_mag + 1e-10), axis=0))
    arith_mean = np.mean(Sxx_mag, axis=0)
    flatness = geo_mean / (arith_mean + 1e-10)
    results["spectral_flatness_mean"] = float(np.mean(flatness))

    return results


def numerical_diff_analysis(model, coreml_transformer, prompt, duration, steps):
    """Compare CoreML vs MPS outputs numerically."""
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # MPS inference
    with torch.no_grad():
        mps_latents = model.model.inference_flow(
            prompt, duration=duration, num_inference_steps=steps,
        )
        if device == "mps":
            torch.mps.synchronize()
        mps_wave = model.vae.decode(mps_latents.transpose(2, 1)).sample.cpu()[0]

    mps_audio = mps_wave.numpy().mean(axis=0).astype(np.float32)

    # CoreML inference (if available)
    if coreml_transformer is None:
        return None

    # Import coreml_inference from generator_worker
    sys.path.insert(0, os.path.join(PROJECT_DIR, "endless_tango"))
    from generator_worker import coreml_inference

    with torch.no_grad():
        coreml_latents = coreml_inference(model, coreml_transformer, prompt, duration, steps)
        coreml_wave = model.vae.decode(coreml_latents.transpose(2, 1)).sample.cpu()[0]

    coreml_audio = coreml_wave.numpy().mean(axis=0).astype(np.float32)

    # Truncate to same length
    min_len = min(len(mps_audio), len(coreml_audio))
    mps_audio = mps_audio[:min_len]
    coreml_audio = coreml_audio[:min_len]

    diff = np.abs(mps_audio - coreml_audio)
    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "correlation": float(np.corrcoef(mps_audio, coreml_audio)[0, 1]),
        "snr_db": float(10 * np.log10(
            np.sum(mps_audio ** 2) / (np.sum(diff ** 2) + 1e-10)
        )),
    }


def benchmark_latency(model, coreml_transformer, backend, prompt, duration, steps, n_runs=10):
    """Measure generation latency with statistics."""
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sys.path.insert(0, os.path.join(PROJECT_DIR, "endless_tango"))
    from generator_worker import coreml_inference

    times = []
    audios = []

    for i in range(n_runs + 2):  # +2 warmup
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

        t0 = time.time()
        with torch.no_grad():
            if backend == "coreml" and coreml_transformer is not None:
                latents = coreml_inference(model, coreml_transformer, prompt, duration, steps)
            else:
                latents = model.model.inference_flow(
                    prompt, duration=duration, num_inference_steps=steps,
                )
            if device == "mps":
                torch.mps.synchronize()
            wave = model.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        elapsed = time.time() - t0

        if i >= 2:  # skip warmup
            times.append(elapsed)
            audio_np = wave.numpy().mean(axis=0).astype(np.float32)
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak * 0.75
            audios.append(audio_np)

        print(f"  [{backend}] Run {i+1}/{n_runs+2}: {elapsed:.3f}s"
              f"{' (warmup)' if i < 2 else ''}", flush=True)

    times = np.array(times)
    return {
        "times": times.tolist(),
        "mean": float(times.mean()),
        "std": float(times.std()),
        "min": float(times.min()),
        "max": float(times.max()),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "rtf_mean": float(times.mean() / duration),
        "rtf_std": float(times.std() / duration),
        "n_runs": n_runs,
    }, audios


def main():
    import torch

    print("=" * 70)
    print("TangoFlux Endless — Paper Benchmark Suite")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS: {torch.backends.mps.is_available()}")
    print(f"Duration: {DURATION}s, Prompts: {len(PROMPTS)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    print("\nLoading TangoFlux...", flush=True)
    from tangoflux import TangoFluxInference
    model = TangoFluxInference(name="declare-lab/TangoFlux", device=device)

    # Load CoreML
    coreml_transformer = None
    coreml_path = os.path.join(COREML_DIR, "transformer.mlpackage")
    if os.path.exists(coreml_path):
        import coremltools as ct
        print("Loading CoreML transformer...", flush=True)
        coreml_transformer = ct.models.MLModel(coreml_path)
        print("CoreML ready.", flush=True)

    all_results = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "device": device,
            "duration_s": DURATION,
            "sample_rate": SAMPLE_RATE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine": os.uname().machine,
            "os": f"{os.uname().sysname} {os.uname().release}",
        },
        "latency": {},
        "step_sweep": {},
        "spectral": {},
        "numerical_diff": {},
        "prompt_variance": {},
    }

    # ============================================================
    # 1. Latency benchmark: CoreML vs MPS, 25 steps, 10 runs each
    # ============================================================
    print("\n" + "=" * 70)
    print("1. LATENCY BENCHMARK (10 runs per backend)")
    print("=" * 70)

    for backend in (["coreml", "mps"] if coreml_transformer else ["mps"]):
        print(f"\n--- {backend.upper()} ---")
        stats, audios = benchmark_latency(
            model, coreml_transformer, backend,
            PROMPTS[0], DURATION, 25, n_runs=10,
        )
        all_results["latency"][backend] = stats
        print(f"  Mean: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"  RTF:  {stats['rtf_mean']:.3f}x ± {stats['rtf_std']:.3f}x")
        print(f"  Range: [{stats['min']:.3f}s, {stats['max']:.3f}s]")
        print(f"  P95: {stats['p95']:.3f}s, P99: {stats['p99']:.3f}s")

    # ============================================================
    # 2. Step count sweep: 10, 15, 20, 25, 30, 50 steps
    # ============================================================
    print("\n" + "=" * 70)
    print("2. STEP COUNT SWEEP (CoreML, 5 runs each)")
    print("=" * 70)

    backend = "coreml" if coreml_transformer else "mps"
    for steps in [10, 15, 20, 25, 30, 50]:
        print(f"\n--- {steps} steps ---")
        stats, audios = benchmark_latency(
            model, coreml_transformer, backend,
            PROMPTS[0], DURATION, steps, n_runs=5,
        )
        all_results["step_sweep"][str(steps)] = stats

        # Spectral analysis on last generated audio
        try:
            from scipy import signal
            spec = spectral_analysis(audios[-1])
            all_results["step_sweep"][str(steps)]["spectral"] = spec
            print(f"  Time: {stats['mean']:.3f}s, RMS: {spec['rms']:.4f}, "
                  f"Centroid: {spec['spectral_centroid_mean']:.0f}Hz, "
                  f"Flatness: {spec['spectral_flatness_mean']:.4f}")
        except ImportError:
            print(f"  Time: {stats['mean']:.3f}s (scipy not available for spectral)")

    # ============================================================
    # 3. Prompt variance: same 5 prompts, measure consistency
    # ============================================================
    print("\n" + "=" * 70)
    print("3. PROMPT VARIANCE (5 prompts, 3 runs each)")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}: '{prompt[:50]}...' ---")
        stats, audios = benchmark_latency(
            model, coreml_transformer, backend,
            prompt, DURATION, 25, n_runs=3,
        )
        result = {"latency": stats}

        try:
            from scipy import signal
            specs = [spectral_analysis(a) for a in audios]
            result["spectral_mean"] = {
                k: float(np.mean([s[k] for s in specs])) for k in specs[0]
            }
            result["spectral_std"] = {
                k: float(np.std([s[k] for s in specs])) for k in specs[0]
            }
            print(f"  Time: {stats['mean']:.3f}s, "
                  f"RMS: {result['spectral_mean']['rms']:.4f} ± {result['spectral_std']['rms']:.4f}")
        except ImportError:
            print(f"  Time: {stats['mean']:.3f}s")

        all_results["prompt_variance"][f"prompt_{i+1}"] = result

    # ============================================================
    # 4. Numerical precision: CoreML vs MPS diff
    # ============================================================
    if coreml_transformer:
        print("\n" + "=" * 70)
        print("4. NUMERICAL PRECISION (CoreML vs MPS)")
        print("=" * 70)

        for i, prompt in enumerate(PROMPTS[:3]):
            print(f"\n--- Prompt {i+1} ---")
            diff = numerical_diff_analysis(model, coreml_transformer, prompt, DURATION, 25)
            if diff:
                all_results["numerical_diff"][f"prompt_{i+1}"] = diff
                print(f"  Max diff: {diff['max_diff']:.6f}")
                print(f"  Mean diff: {diff['mean_diff']:.6f}")
                print(f"  Correlation: {diff['correlation']:.6f}")
                print(f"  SNR: {diff['snr_db']:.1f} dB")

    # ============================================================
    # Save results
    # ============================================================
    results_path = os.path.join(RESULTS_DIR, "paper_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)

    if "coreml" in all_results["latency"] and "mps" in all_results["latency"]:
        cml = all_results["latency"]["coreml"]
        mps = all_results["latency"]["mps"]
        print(f"\n{'Backend':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P95':>8} {'RTF':>8}")
        print("-" * 66)
        print(f"{'MPS':<12} {mps['mean']:>7.3f}s {mps['std']:>7.3f}s {mps['min']:>7.3f}s "
              f"{mps['max']:>7.3f}s {mps['p95']:>7.3f}s {mps['rtf_mean']:>7.3f}x")
        print(f"{'CoreML':<12} {cml['mean']:>7.3f}s {cml['std']:>7.3f}s {cml['min']:>7.3f}s "
              f"{cml['max']:>7.3f}s {cml['p95']:>7.3f}s {cml['rtf_mean']:>7.3f}x")
        print(f"\nSpeedup: {mps['mean']/cml['mean']:.2f}x")

    print("\nStep sweep:")
    print(f"{'Steps':>6} {'Time':>8} {'RTF':>8}")
    print("-" * 26)
    for steps in sorted(all_results["step_sweep"].keys(), key=int):
        s = all_results["step_sweep"][steps]
        print(f"{steps:>6} {s['mean']:>7.3f}s {s['rtf_mean']:>7.3f}x")


if __name__ == "__main__":
    main()
