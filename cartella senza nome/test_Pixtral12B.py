import os
import random
import json
import base64
import csv
import argparse
from pathlib import Path
from vllm import LLM
from vllm.sampling_params import SamplingParams

# ── Argomenti ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="structured", choices=["structured", "unstructured"])
args = parser.parse_args()

# ── Configurazione ambiente ──────────────────────────────
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

After reasoning internally, output the final decision in the following format:

REAL
or
FAKE

Do not output anything else after this line."""

PROMPT = PROMPT_STRUCTURED if args.prompt == "structured" else PROMPT_UNSTRUCTURED
print(f"Usando prompt: {args.prompt}")

# ── Config ──────────────────────────────────────────────
DATASET_ROOT      = Path("/work/dataset")
CATEGORIES        = ["original", "redigital", "transfer"]
SAMPLES_PER_CLASS = 1000
METRICS_FILE      = f"/work/project/metrics_pixtral12b_{args.prompt}.json"
RESULTS_FILE      = f"/work/project/results_pixtral12b_{args.prompt}.csv"
SEED              = 42

# Token IDs per FAKE e REAL nel vocabolario di Pixtral
# fake=115786, real=15074, Fake=106309, Real=24877
ALLOWED_TOKEN_IDS = [115786, 15074, 106309, 24877]

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
def run_inference(llm, samples, sampling_params):
    results = []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"Progresso: {i}/{len(samples)}")

        with open(sample["path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext  = Path(sample["path"]).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]
        }]

        output = llm.chat(messages, sampling_params)
        raw    = output[0].outputs[0].text.strip().upper()

        if "FAKE" in raw:
            prediction = "FAKE"
        elif "REAL" in raw:
            prediction = "REAL"
        else:
            prediction = "UNKNOWN"

        results.append({
            "category":     sample["category"],
            "ground_truth": sample["ground_truth"],
            "prediction":   prediction,
            "correct":      prediction == sample["ground_truth"]
        })

    return results

# ── Metriche ─────────────────────────────────────────────
def compute_metrics(results):
    metrics = {}
    for category in CATEGORIES + ["overall"]:
        subset  = results if category == "overall" \
                  else [r for r in results if r["category"] == category]
        total   = len(subset)

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
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "fpr":       round(fpr, 4),
            "fnr":       round(fnr, 4),
            "confusion_matrix": {
                "TP": tp, "FP": fp,
                "TN": tn, "FN": fn
            }
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
        for r in results:
            writer.writerow({
                "category":     r["category"],
                "ground_truth": r["ground_truth"],
                "prediction":   r["prediction"],
                "correct":      r["correct"]
            })
    print(f"Risultati salvati in {RESULTS_FILE}")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Caricamento modello...")
    llm = LLM(
        model="mistralai/Pixtral-12B-2409",
        tokenizer_mode="mistral",
        max_model_len=23568,
        download_dir="/work/models",
        gpu_memory_utilization=0.95
    )
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        allowed_token_ids=ALLOWED_TOKEN_IDS
    )

    samples = load_samples()
    results = run_inference(llm, samples, sampling_params)
    metrics = compute_metrics(results)
    save_results(results, metrics)