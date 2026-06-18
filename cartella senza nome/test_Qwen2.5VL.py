import os
import random
import json
import csv
import argparse
import time
import warnings
import logging
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.manual_seed(9999)
set_seed(9999)

# ── Argomenti ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="unstructured", choices=["structured", "unstructured"])
parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "32B", "72B"])
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GPU_CONFIG = {
    "7B":  "0",
    "32B": "0,1,2",
    "72B": "0,1,2,3",
}
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CONFIG[args.model_size]
print(f"Usando GPU: {GPU_CONFIG[args.model_size]}")

MODEL_MAP = {
    "7B":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "32B": "Qwen/Qwen2.5-VL-32B-Instruct",
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
}

DATASET_ROOT      = Path("/work/dataset")
CATEGORIES        = ["original", "redigital", "transfer"]
SAMPLES_PER_CLASS = 1000
METRICS_FILE      = f"/work/project/metrics_qwen_{args.model_size}_{args.prompt}.json"
RESULTS_FILE      = f"/work/project/results_qwen_{args.model_size}_{args.prompt}.csv"
MODEL_PATH        = MODEL_MAP[args.model_size]

print(f"Usando modello: {MODEL_PATH}")
print(f"Usando prompt: {args.prompt}")

# ── Prompt ───────────────────────────────────────────────
PROMPT_UNSTRUCTURED = """Look at this image and determine whether it is a real photograph or an AI-generated image. The image may contain any subject: people, animals, objects, landscapes, or urban scenes.
Consider any visual inconsistencies you notice and answer with only one word.
Do not explain your reasoning.
Do not output anything except the final classification.

Answer with REAL or FAKE only."""

PROMPT_STRUCTURED = """You are an expert in visual forensics specialized in detecting AI-generated and manipulated images under real-world conditions (including compression and re-digitization artifacts).
Carefully analyze the image using the following criteria:

1. Texture and Fine Details
   - Look for unnatural smoothness, over-sharpening, or inconsistent noise.
   - Check for missing or distorted fine structures (edges, text, small objects).

2. Global Lighting and Physical Consistency
   - Verify consistent lighting direction across the entire scene.
   - Check shadows, reflections, and illumination coherence between objects.

3. Structural and Semantic Coherence
   - Analyze whether objects, people, and scene elements are physically and logically plausible.
   - Look for distorted shapes, impossible structures, or inconsistent geometry.

4. Background and Scene Integrity
   - Detect repeating patterns, warped regions, unnatural transitions, or blending artifacts.
   - Check text, signs, and detailed areas for corruption or inconsistency.

5. Robustness to Real-World Degradation
   - Consider whether artifacts could come from compression or re-capturing.
   - Distinguish between real degradation and synthetic generation artifacts.

6. Overall Realism
   - Evaluate whether the image looks like a natural photograph captured in the real world.

After analyzing the image internally, provide your judgment using only one word:
- Output REAL if the image is a natural photograph.
- Output FAKE if the image appears AI-generated.

Answer:"""

PROMPT = PROMPT_STRUCTURED if args.prompt == "structured" else PROMPT_UNSTRUCTURED


# ── Carica campioni ──────────────────────────────────────
def load_samples():
    samples = []
    for category in CATEGORIES:
        for label, folder in [("FAKE", "ai"), ("REAL", "real")]:
            path = DATASET_ROOT / category / folder
            images = list(path.glob("*.jpg")) + \
                     list(path.glob("*.png")) + \
                     list(path.glob("*.jpeg"))
            selected = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
            for img_path in selected:
                samples.append({
                    "path":         str(img_path),
                    "category":     category,
                    "ground_truth": label
                })
    print(f"Totale immagini: {len(samples)}")
    return samples


# ── Inferenza ────────────────────────────────────────────
def run_inference(model, processor, samples):
    results = []
    real_count = fake_count = unknown_count = error_count = 0
    start_time = time.time()

    progress = tqdm(samples, desc="Classificazione", dynamic_ncols=True)

    for sample in progress:
        try:
            img = Image.open(sample["path"]).convert("RGB")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  PROMPT},
            ]}]

            chat = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs = process_vision_info(messages)[0]

            inputs = processor(
                text=[chat],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.0
                )

            text   = processor.decode(output[0], skip_special_tokens=True)
            answer = text.split("Answer:")[-1].strip()

            if "REAL" in answer:
                prediction = "REAL"
                real_count += 1
            elif "FAKE" in answer:
                prediction = "FAKE"
                fake_count += 1
            else:
                prediction = "UNKNOWN"
                unknown_count += 1
                progress.write(f"UNKNOWN output per {Path(sample['path']).name}: '{answer}'")

            results.append({
                "category":     sample["category"],
                "ground_truth": sample["ground_truth"],
                "prediction":   prediction,
                "correct":      prediction == sample["ground_truth"]
            })

        except Exception as e:
            error_count += 1
            progress.write(f"ERROR: {Path(sample['path']).name}: {e}")
            results.append({
                "category":     sample["category"],
                "ground_truth": sample["ground_truth"],
                "prediction":   "ERROR",
                "correct":      False
            })

        progress.set_postfix({
            "REAL": real_count,
            "FAKE": fake_count,
            "UNK":  unknown_count,
            "ERR":  error_count,
        })

    elapsed = time.time() - start_time
    print(f"\nTempo totale: {elapsed:.0f}s ({elapsed/len(samples):.1f}s/img)")
    return results


# ── Metriche ─────────────────────────────────────────────
def compute_metrics(results):
    metrics = {}
    for category in CATEGORIES + ["overall"]:
        subset = results if category == "overall" \
                 else [r for r in results if r["category"] == category]
        subset = [r for r in subset if r["prediction"] not in ["ERROR"]]
        total = len(subset)
        if not subset:
            continue

        tp = sum(1 for r in subset if r["prediction"] == "FAKE" and r["ground_truth"] == "FAKE")
        fp = sum(1 for r in subset if r["prediction"] == "FAKE" and r["ground_truth"] == "REAL")
        tn = sum(1 for r in subset if r["prediction"] == "REAL" and r["ground_truth"] == "REAL")
        fn = sum(1 for r in subset if r["prediction"] == "REAL" and r["ground_truth"] == "FAKE")

        accuracy  = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp)    if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr       = fp / (fp + tn)    if (fp + tn) > 0 else 0
        fnr       = fn / (fn + tp)    if (fn + tp) > 0 else 0

        metrics[category] = {
            "total":     total,
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "fpr":       round(fpr,       4),
            "fnr":       round(fnr,       4),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        }

        print(f"[{category.upper():10}] "
              f"Acc: {accuracy:.3f} | F1: {f1:.3f} | "
              f"FPR: {fpr:.3f} | FNR: {fnr:.3f} | "
              f"Total: {total}")

    return metrics


# ── Salvataggio ──────────────────────────────────────────
def save_results(results, metrics):
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriche salvate in {METRICS_FILE}")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Risultati salvati in {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GPU disponibili: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")

    print("\nCaricamento modello...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/work/models"
    ).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir="/work/models"
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {total_params/1e9:.2f}B")
    if hasattr(model, "hf_device_map"):
        devices_used = sorted(set(str(v) for v in model.hf_device_map.values()))
        print(f"Device usati: {devices_used}")
    for g in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(g) / 1024**3
        print(f"VRAM GPU {g}: {allocated:.2f}GB")

    # ── Quick test ────────────────────────────────────────
    print("\nQuick test...")
    test_messages = [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the single word: REAL"}
    ]}]
    test_chat   = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = processor(text=[test_chat], return_tensors="pt").to(model.device)
    with torch.no_grad():
        test_output = model.generate(
            **test_inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    test_text   = processor.decode(test_output[0], skip_special_tokens=True)
    test_answer = test_text.split("assistant")[-1].strip()
    print(f"Quick test answer: '{test_answer}'")
    if "REAL" not in test_answer.upper():
        print("WARNING: quick test non ha risposto REAL — verifica la configurazione.")
        raise SystemExit(1)
    else:
        print("OK: quick test superato.\n")

    # ── Pipeline principale ───────────────────────────────
    samples = load_samples()
    results = run_inference(model, processor, samples)
    metrics = compute_metrics(results)
    save_results(results, metrics)
    print("\nCompletato!")
