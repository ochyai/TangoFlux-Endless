#!/usr/bin/env python3
"""
TangoFlux Generator Worker - CoreML高速化版
- transformerをCoreML（ANE/GPU）で実行、T5とVAEはPyTorch
- 25ステップで約3.5秒（MPS比2倍高速）
- メインプロセスと .npy ファイル経由で通信
- 生成したクリップをアーカイブに保存（エンベディング付き、上限200件）
"""

import sys
import os
import json
import time
import hashlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(SCRIPT_DIR, "clip_archive")
COREML_DIR = os.path.join(SCRIPT_DIR, "coreml_models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
MAX_ARCHIVE_SIZE = 200
MAX_LOG_FILES = 10  # ログファイル上限
MAX_LOG_LINES = 5000  # 1ファイルあたりの行数上限


def setup_logging():
    """ログファイルの設定。古いログの自動クリーンアップ"""
    import logging
    from datetime import datetime
    os.makedirs(LOG_DIR, exist_ok=True)

    # 古いログファイルを削除
    log_files = sorted(
        [f for f in os.listdir(LOG_DIR) if f.startswith("worker_") and f.endswith(".log")],
    )
    if len(log_files) >= MAX_LOG_FILES:
        for old_log in log_files[:len(log_files) - MAX_LOG_FILES + 1]:
            try:
                os.remove(os.path.join(LOG_DIR, old_log))
            except OSError:
                pass

    log_filename = f"worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger("worker")
    logger.setLevel(logging.INFO)
    # ファイルハンドラ
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    # stdout（メインプロセスのモニター用）
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    logger.info(f"WORKER: Log file: {log_path}")
    return logger


def log(msg):
    """stdoutとログファイルに同時出力"""
    _logger.info(msg)


_logger = None


def load_embedding_model():
    from transformers import AutoTokenizer, AutoModel
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    log("WORKER: Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emb_model = AutoModel.from_pretrained(model_name)
    emb_model.eval()
    log("WORKER: Embedding model ready")
    return tokenizer, emb_model


def compute_embedding(text, tokenizer, emb_model):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = emb_model(**inputs)
    emb = output.last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def save_to_archive(audio_np, prompt, embedding, duration):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    clip_id = hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:12]
    clip_filename = f"{clip_id}.npy"
    np.save(os.path.join(ARCHIVE_DIR, clip_filename), audio_np)

    index_path = os.path.join(ARCHIVE_DIR, "index.json")
    index = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = []

    index.append({
        "id": clip_id, "filename": clip_filename, "prompt": prompt,
        "embedding": embedding.tolist(), "duration": duration, "timestamp": time.time(),
    })

    if len(index) > MAX_ARCHIVE_SIZE:
        index.sort(key=lambda x: x.get("timestamp", 0))
        to_remove = index[:len(index) - MAX_ARCHIVE_SIZE]
        index = index[len(index) - MAX_ARCHIVE_SIZE:]
        for entry in to_remove:
            try:
                os.remove(os.path.join(ARCHIVE_DIR, entry["filename"]))
            except OSError:
                pass
        log(f"WORKER: Archive cleanup: removed {len(to_remove)} old clips")

    tmp_index = index_path + ".tmp"
    with open(tmp_index, "w") as f:
        json.dump(index, f)
    os.rename(tmp_index, index_path)
    return clip_id


def dir_exists(path):
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def safe_listdir(path):
    try:
        return os.listdir(path)
    except (OSError, FileNotFoundError):
        return []


def coreml_inference(model, coreml_transformer, prompt, duration, num_steps):
    """CoreMLのtransformerを使ったカスタム推論ループ"""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = model.model.transformer.device
    max_text_len = 64  # CoreML固定シェイプに合わせる

    # T5テキストエンコーディング（CFG用: conditional + unconditional）
    with torch.no_grad():
        # conditional
        batch = model.model.tokenizer(
            [prompt], max_length=max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        cond_embeds = model.model.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        cond_mask = (attention_mask == 1).to(device)

        # unconditional
        uncond_batch = model.model.tokenizer(
            [""], max_length=max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        uncond_ids = uncond_batch.input_ids.to(device)
        uncond_mask_raw = uncond_batch.attention_mask.to(device)
        uncond_embeds = model.model.text_encoder(
            input_ids=uncond_ids, attention_mask=uncond_mask_raw
        )[0]
        uncond_mask = (uncond_mask_raw == 1).to(device)

        # CFGバッチ結合
        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds])  # (2, 64, 1024)
        boolean_mask = torch.cat([uncond_mask, cond_mask])  # (2, 64)

        # Duration
        dur_tensor = torch.tensor([duration], device=device)
        duration_hidden_states = model.model.encode_duration(dur_tensor)  # (1, 1, 1024)
        duration_hidden_states = duration_hidden_states.repeat(2, 1, 1)  # (2, 1, 1024)

        # Pooled projection
        mask_expanded = boolean_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        masked_data = torch.where(
            mask_expanded, encoder_hidden_states, torch.tensor(float("nan"))
        )
        pooled = torch.nanmean(masked_data, dim=1)  # (2, 1024)
        pooled_projection = model.model.fc(pooled)  # (2, 1024)

        # encoder_hidden_states + duration
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, duration_hidden_states], dim=1
        )  # (2, 65, 1024)

    # スケジューラ設定
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    from tangoflux.model import retrieve_timesteps
    timesteps, num_steps = retrieve_timesteps(scheduler, num_steps, device, sigmas=sigmas)

    # 初期ノイズ
    audio_seq_len = model.model.audio_seq_len  # 645
    latents = torch.randn(1, audio_seq_len, 64)

    # ID tensors
    txt_ids = torch.zeros(2, encoder_hidden_states.shape[1], 3)
    img_ids = torch.arange(audio_seq_len).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 3).float()

    # numpy変換（CoreML入力用）
    enc_np = encoder_hidden_states.cpu().numpy().astype(np.float16)
    pooled_np = pooled_projection.cpu().numpy().astype(np.float16)
    txt_ids_np = txt_ids.numpy().astype(np.float16)
    img_ids_np = img_ids.numpy().astype(np.float16)

    latents = latents.to(device)
    timesteps = timesteps.to(device)
    guidance_scale = 3.0

    # デノイジングループ（CoreML transformer）
    for t in timesteps:
        latents_input = torch.cat([latents, latents])  # CFG: (2, 645, 64)
        hidden_np = latents_input.cpu().numpy().astype(np.float16)
        timestep_np = np.array([t.item() / 1000], dtype=np.float16)

        pred = coreml_transformer.predict({
            "hidden_states": hidden_np,
            "timestep": timestep_np,
            "pooled_projections": pooled_np,
            "encoder_hidden_states": enc_np,
            "txt_ids": txt_ids_np,
            "img_ids": img_ids_np,
        })

        noise_pred = torch.from_numpy(
            pred["noise_pred"].astype(np.float32)
        ).to(device)

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def main():
    global _logger

    if len(sys.argv) < 5:
        print("Usage: generator_worker.py <output_dir> <prompt> <duration> <steps>", file=sys.stderr)
        sys.exit(1)

    _logger = setup_logging()

    output_dir = sys.argv[1]
    prompt = sys.argv[2]
    duration = int(sys.argv[3])
    steps = int(sys.argv[4])
    cmd_file = os.path.join(output_dir, "command.json")

    import torch
    from tangoflux import TangoFluxInference

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"WORKER: Loading TangoFlux on {device.upper()}...")
    model = TangoFluxInference(name="declare-lab/TangoFlux", device=device)

    # CoreMLモデルをロード（存在する場合）
    coreml_transformer = None
    coreml_path = os.path.join(COREML_DIR, "transformer.mlpackage")
    if os.path.exists(coreml_path):
        try:
            import coremltools as ct
            log("WORKER: Loading CoreML transformer...")
            coreml_transformer = ct.models.MLModel(coreml_path)
            log("WORKER: CoreML transformer ready (ANE/GPU accelerated)")
        except Exception as e:
            log(f"WORKER: CoreML load failed ({e}), using MPS fallback")

    if coreml_transformer:
        log("WORKER: TangoFlux ready - CoreML transformer + MPS VAE")
    else:
        log(f"WORKER: TangoFlux ready on {device.upper()} (MPS only)")

    emb_tokenizer, emb_model = load_embedding_model()

    current_prompt = prompt
    current_duration = duration
    current_steps = steps
    current_embedding = compute_embedding(current_prompt, emb_tokenizer, emb_model)

    if dir_exists(output_dir):
        try:
            np.save(os.path.join(output_dir, "current_embedding.npy"), current_embedding)
        except OSError:
            pass

    gen_count = 0

    while True:
        if not dir_exists(output_dir):
            log("WORKER: Output dir removed, exiting")
            break

        if os.path.exists(cmd_file):
            try:
                with open(cmd_file, "r") as f:
                    cmd = json.load(f)
                if cmd.get("action") == "stop":
                    log("WORKER: Stop command received")
                    break
                if cmd.get("action") == "update":
                    current_prompt = cmd.get("prompt", current_prompt)
                    current_duration = cmd.get("duration", current_duration)
                    current_steps = cmd.get("steps", current_steps)
                    current_embedding = compute_embedding(current_prompt, emb_tokenizer, emb_model)
                    try:
                        np.save(os.path.join(output_dir, "current_embedding.npy"), current_embedding)
                    except OSError:
                        pass
                    log(f"WORKER: Prompt updated: {current_prompt[:50]}...")
                os.remove(cmd_file)
            except (json.JSONDecodeError, OSError):
                pass

        pending = len([f for f in safe_listdir(output_dir)
                       if f.startswith("clip_") and f.endswith(".npy") and not f.endswith(".tmp")])
        if pending >= 3:
            time.sleep(0.5)
            continue

        gen_count += 1
        backend = "CoreML" if coreml_transformer else "MPS"
        log(f"WORKER: Generating #{gen_count} [{backend}] (steps={current_steps})...")

        t0 = time.time()
        try:
            with torch.no_grad():
                if coreml_transformer:
                    latents = coreml_inference(
                        model, coreml_transformer,
                        current_prompt, current_duration, current_steps,
                    )
                else:
                    latents = model.model.inference_flow(
                        current_prompt,
                        duration=current_duration,
                        num_inference_steps=current_steps,
                    )
                if device == "mps":
                    torch.mps.synchronize()
                wave = model.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]

            waveform_end = int(current_duration * model.vae.config.sampling_rate)
            wave = wave[:, :waveform_end]
            audio_np = wave.numpy().mean(axis=0).astype(np.float32)
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak * 0.75

            elapsed = time.time() - t0

            if dir_exists(output_dir):
                clip_path = os.path.join(output_dir, f"clip_{gen_count:04d}.npy")
                tmp_path = clip_path + ".tmp"
                try:
                    np.save(tmp_path, audio_np)
                    os.rename(tmp_path, clip_path)
                except OSError:
                    log("WORKER: Output dir gone, skipping write")
            else:
                log("WORKER: Output dir removed, exiting")
                break

            clip_emb = compute_embedding(current_prompt, emb_tokenizer, emb_model)
            archive_id = save_to_archive(audio_np, current_prompt, clip_emb, current_duration)
            log(f"WORKER: #{gen_count} saved ({elapsed:.1f}s, RTF {elapsed/current_duration:.2f}x) [archived:{archive_id}]")

        except Exception as e:
            log(f"WORKER: Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)

    log("WORKER: Exiting")


if __name__ == "__main__":
    main()
