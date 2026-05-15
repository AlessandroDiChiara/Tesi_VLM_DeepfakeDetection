import os
import random
import json
import csv
import math
import time
import argparse
import warnings
import logging
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.manual_seed(9999)

# ── Argomenti ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="structured", choices=["structured", "unstructured"])
parser.add_argument("--model_size", type=str, default="8B", choices=["8B", "38B"])
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GPU_CONFIG = {
    "8B":  "0",
    "38B": "0,1,2",
}
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CONFIG[args.model_size]
print(f"Usando GPU: {GPU_CONFIG[args.model_size]}")

MODEL_MAP = {
    "8B":  "OpenGVLab/InternVL3_5-8B-Instruct",
    "38B": "OpenGVLab/InternVL3_5-38B-Instruct",
}

DATASET_ROOT      = Path("/work/dataset")
CATEGORIES        = ["original", "redigital", "transfer"]
SAMPLES_PER_CLASS = 1000
METRICS_FILE      = f"/work/project/metrics_internvl_{args.model_size}_{args.prompt}.json"
RESULTS_FILE      = f"/work/project/results_internvl_{args.model_size}_{args.prompt}.csv"
SEED              = 42
MODEL_PATH        = MODEL_MAP[args.model_size]

print(f"Usando modello: {MODEL_PATH}")
print(f"Usando prompt: {args.prompt}")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Prompt ───────────────────────────────────────────────
PROMPT_UNSTRUCTURED = """<image>\nLook at this image and determine whether it is a real photograph or an AI-generated image. The image may contain any subject: people, animals, objects, landscapes, or urban scenes.
Consider any visual inconsistencies you notice and answer with only one word.

Do not explain your reasoning.
Do not output anything except the final classification.

Answer with REAL or FAKE only."""

PROMPT_STRUCTURED = """<image>\nYou are an expert in visual forensics specialized in detecting AI-generated and manipulated images under real-world conditions (including compression and re-digitization artifacts).

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

After reasoning internally, output ONLY one word:
- Output REAL if the image is a natural photograph.
- Output FAKE if the image appears AI-generated.

Answer:"""

PROMPT = PROMPT_STRUCTURED if args.prompt == "structured" else PROMPT_UNSTRUCTURED

# ── Preprocessing ────────────────────────────────────────
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calcola aspect ratio esistente e trova il più vicino tra quelli ottenibili con blocchi di dimensione image_size
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    #resize l'immagine al nuovo aspect ratio   
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_path, input_size=448, max_num=9):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# ── Carica campioni ──────────────────────────────────────
def load_samples():
    random.seed(SEED)
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
def run_inference(model, tokenizer, samples):
    generation_config = dict(max_new_tokens=5, do_sample=False)
    results = []
    real_count = fake_count = unknown_count = error_count = 0

    progress = tqdm(samples, desc="Classificazione", dynamic_ncols=True)

    for sample in progress:
        try:
            pixel_values = load_image(sample["path"]).to(torch.bfloat16).to(next(model.parameters()).device)
            response = model.chat(tokenizer, pixel_values, PROMPT, generation_config)
            torch.cuda.empty_cache()

            raw = response.strip().split()[0].upper() if response.strip() else "UNKNOWN"

            if "FAKE" in raw:
                prediction = "FAKE"
                fake_count += 1
            elif "REAL" in raw:
                prediction = "REAL"
                real_count += 1
            else:
                prediction = "UNKNOWN"
                unknown_count += 1
                progress.write(f"UNKNOWN: {Path(sample['path']).name}: '{response.strip()}'")

            results.append({
                "category":     sample["category"],
                "ground_truth": sample["ground_truth"],
                "prediction":   prediction,
                "correct":      prediction == sample["ground_truth"]
            })

        except Exception as e:
            error_count += 1
            progress.write(f"ERROR: {Path(sample['path']).name}: {e}")
            torch.cuda.empty_cache()
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

    return results

# ── Metriche ─────────────────────────────────────────────
def compute_metrics(results):
    metrics = {}
    for category in CATEGORIES + ["overall"]:
        subset = results if category == "overall" \
                 else [r for r in results if r["category"] == category]
        subset = [r for r in subset if r["prediction"] != "ERROR"]
        total  = len(subset)
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
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
        cache_dir="/work/models"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        cache_dir="/work/models"
    )
    print("Modello caricato!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {total_params/1e9:.2f}B")
    for g in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(g) / 1024**3
        print(f"VRAM GPU {g}: {allocated:.2f}GB")

    print("\nQuick test...")
    test_img_path = next(Path("/work/dataset").rglob("*.jpg"))
    test_pixels = load_image(str(test_img_path), max_num=4).to(torch.bfloat16).to(next(model.parameters()).device)
    test_response = model.chat(
        tokenizer,
        test_pixels,
        PROMPT,
        dict(max_new_tokens=5, do_sample=False)
    )
    print(f"Quick test answer: '{test_response.strip()}'")
    if test_response.strip().split()[0].upper() not in ["REAL", "FAKE"]:
        print("WARNING: quick test non ha risposto REAL o FAKE.")
    else:
        print("OK: quick test superato.\n")

    samples = load_samples()
    results = run_inference(model, tokenizer, samples)
    metrics = compute_metrics(results)
    save_results(results, metrics)
    print("\nCompletato!")
