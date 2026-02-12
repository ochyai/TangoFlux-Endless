#!/usr/bin/env python3
"""
TangoFlux FluxTransformer2DModel → CoreML変換スクリプト
- transformerのみ変換（最大ボトルネック）
- T5エンコーダとVAEはPyTorch MPS/CPUのまま
- 固定入力シェイプ（ANE最適化のため）
"""

import os
import sys
import time
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COREML_DIR = os.path.join(SCRIPT_DIR, "coreml_models")


class TransformerWrapper(torch.nn.Module):
    """CoreML変換用のラッパー。全入力をfloat32に統一し、
    guidance=None, return_dict=False を固定"""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, pooled_projections,
                encoder_hidden_states, txt_ids, img_ids):
        result = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=None,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        return result[0]


def convert_transformer():
    """FluxTransformer2DModelをCoreMLに変換"""
    import coremltools as ct
    from tangoflux import TangoFluxInference

    print("Loading TangoFlux model...", flush=True)
    model = TangoFluxInference(name="declare-lab/TangoFlux", device="cpu")

    wrapper = TransformerWrapper(model.model.transformer)
    wrapper.eval()

    # 固定入力シェイプ（CFG時: batch=2, audio_seq=645, text_seq=65）
    batch_size = 2
    audio_seq_len = 645
    text_seq_len = 65  # max_text_seq_len(64) + duration(1)
    in_channels = 64
    joint_dim = 1024

    print("Creating sample inputs...", flush=True)
    sample_inputs = {
        "hidden_states": torch.randn(batch_size, audio_seq_len, in_channels),
        "timestep": torch.tensor([0.5]),
        "pooled_projections": torch.randn(batch_size, joint_dim),
        "encoder_hidden_states": torch.randn(batch_size, text_seq_len, joint_dim),
        "txt_ids": torch.zeros(batch_size, text_seq_len, 3),
        "img_ids": torch.zeros(batch_size, audio_seq_len, 3),
    }

    # torch.jit.trace
    print("Tracing model...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (
                sample_inputs["hidden_states"],
                sample_inputs["timestep"],
                sample_inputs["pooled_projections"],
                sample_inputs["encoder_hidden_states"],
                sample_inputs["txt_ids"],
                sample_inputs["img_ids"],
            ),
        )

    # 精度検証
    print("Verifying trace accuracy...", flush=True)
    with torch.no_grad():
        orig_out = wrapper(**sample_inputs)
        traced_out = traced(
            sample_inputs["hidden_states"],
            sample_inputs["timestep"],
            sample_inputs["pooled_projections"],
            sample_inputs["encoder_hidden_states"],
            sample_inputs["txt_ids"],
            sample_inputs["img_ids"],
        )
    max_diff = (orig_out - traced_out).abs().max().item()
    print(f"Trace max diff: {max_diff:.2e}", flush=True)
    if max_diff > 1e-4:
        print("WARNING: Trace accuracy is low, conversion may produce artifacts", flush=True)

    # CoreML変換
    print("Converting to CoreML (this may take a few minutes)...", flush=True)
    t0 = time.time()

    input_shapes = [
        ct.TensorType(
            name="hidden_states",
            shape=(batch_size, audio_seq_len, in_channels),
        ),
        ct.TensorType(
            name="timestep",
            shape=(1,),
        ),
        ct.TensorType(
            name="pooled_projections",
            shape=(batch_size, joint_dim),
        ),
        ct.TensorType(
            name="encoder_hidden_states",
            shape=(batch_size, text_seq_len, joint_dim),
        ),
        ct.TensorType(
            name="txt_ids",
            shape=(batch_size, text_seq_len, 3),
        ),
        ct.TensorType(
            name="img_ids",
            shape=(batch_size, audio_seq_len, 3),
        ),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=input_shapes,
        outputs=[ct.TensorType(name="noise_pred")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    elapsed = time.time() - t0
    print(f"CoreML conversion done in {elapsed:.1f}s", flush=True)

    # 保存
    os.makedirs(COREML_DIR, exist_ok=True)
    model_path = os.path.join(COREML_DIR, "transformer.mlpackage")
    mlmodel.save(model_path)
    print(f"Saved to: {model_path}", flush=True)

    # CoreMLモデルの推論テスト
    print("Testing CoreML inference...", flush=True)
    t0 = time.time()
    coreml_inputs = {
        "hidden_states": sample_inputs["hidden_states"].numpy().astype(np.float16),
        "timestep": sample_inputs["timestep"].numpy().astype(np.float16),
        "pooled_projections": sample_inputs["pooled_projections"].numpy().astype(np.float16),
        "encoder_hidden_states": sample_inputs["encoder_hidden_states"].numpy().astype(np.float16),
        "txt_ids": sample_inputs["txt_ids"].numpy().astype(np.float16),
        "img_ids": sample_inputs["img_ids"].numpy().astype(np.float16),
    }

    # ウォームアップ
    for _ in range(2):
        pred = mlmodel.predict(coreml_inputs)

    # ベンチマーク（25ステップ分シミュレート）
    t0 = time.time()
    for _ in range(25):
        pred = mlmodel.predict(coreml_inputs)
    coreml_time = time.time() - t0

    # MPS比較
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper_mps = TransformerWrapper(model.model.transformer.to(device))
    wrapper_mps.eval()
    mps_inputs = {k: v.to(device) for k, v in sample_inputs.items()}

    # MPS ウォームアップ
    with torch.no_grad():
        for _ in range(2):
            wrapper_mps(**mps_inputs)
        if device == "mps":
            torch.mps.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(25):
            wrapper_mps(**mps_inputs)
        if device == "mps":
            torch.mps.synchronize()
    mps_time = time.time() - t0

    print(f"\n=== 25-step benchmark ===", flush=True)
    print(f"CoreML: {coreml_time:.2f}s ({coreml_time/25*1000:.0f}ms/step)", flush=True)
    print(f"MPS:    {mps_time:.2f}s ({mps_time/25*1000:.0f}ms/step)", flush=True)
    print(f"Speedup: {mps_time/coreml_time:.2f}x", flush=True)

    # 精度比較
    with torch.no_grad():
        mps_out = wrapper_mps(**mps_inputs).cpu()
    coreml_out = torch.from_numpy(pred["noise_pred"].astype(np.float32))
    diff = (mps_out - coreml_out).abs().max().item()
    print(f"Max diff (MPS vs CoreML): {diff:.4f}", flush=True)


if __name__ == "__main__":
    convert_transformer()
