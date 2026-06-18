"""

Output:
    /work/project/pipeline/metrics_zeroshot_32B.json
    /work/project/pipeline/results_zeroshot_32B.csv
"""

import os
import json
import csv
import argparse
import time
import warnings
import logging
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.manual_seed(9999)
set_seed(9999)

# ── Argomenti ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="32B", choices=["7B", "32B", "72B"])
parser.add_argument("--input_json", type=str, default="/work/project/pipeline/final_for_judge.json",
                    help="JSON dalla pipeline — usato solo per la lista delle immagini (path + label_gt)")
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────
os.environ['MPLCONFIGDIR']            = "/work/project"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GPU_CONFIG = {
    "7B":  "0",
    "32B": "0,1,2",
    "72B": "0,1,2,3,4",
}
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CONFIG[args.model_size]
print(f"Usando GPU:    {GPU_CONFIG[args.model_size]}")

MODEL_MAP = {
    "7B":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "32B": "Qwen/Qwen2.5-VL-32B-Instruct",
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
}

MODEL_PATH   = MODEL_MAP[args.model_size]
DATASET_ROOT = "/work/dataset/RRDataset_final"
METRICS_FILE = f"/work/project/pipeline/metrics_zeroshot_{args.model_size}.json"
RESULTS_FILE = f"/work/project/pipeline/results_zeroshot_{args.model_size}.csv"

print(f"Usando modello: {MODEL_PATH}")
print(f"Input JSON:     {args.input_json}")

# ── Prompt strutturato
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


# ── Carica lista immagini dal JSON della pipeline ────────
def load_samples(input_json):
    """
    Legge il JSON finale della pipeline.
    Usa solo path e label_gt — ignora tutto il resto 
    """
    with open(input_json) as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Ricostruisce il path assoluto nel container
        path_server = item.get("path_server",
                      os.path.join(DATASET_ROOT, item["path"]))
        samples.append({
            "path":        path_server,
            "path_rel":    item["path"],
            "category_gt": item["category_gt"],
            "label_gt":    item["label_gt"],
        })

    print(f"Totale immagini: {len(samples)}")
    return samples


# ── Inferenza ────────────────────────────────────────────
def run_inference(model, processor, samples):
    results = []
    real_count = fake_count = unknown_count = error_count = 0
    start_time = time.time()

    progress = tqdm(samples, desc=f"Qwen-{args.model_size} zero-shot", dynamic_ncols=True)

    for sample in progress:
        try:
            img = Image.open(sample["path"]).convert("RGB")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  PROMPT_STRUCTURED},
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
                    temperature=0.0,
                )

            text   = processor.decode(output[0], skip_special_tokens=True)
            answer = text.split("Answer:")[-1].strip()

            if "REAL" in answer.upper():
                prediction = "REAL"
                real_count += 1
            elif "FAKE" in answer.upper():
                prediction = "FAKE"
                fake_count += 1
            else:
                prediction = "UNKNOWN"
                unknown_count += 1
                progress.write(f"UNKNOWN: {Path(sample['path']).name}: '{answer}'")

            results.append({
                "path":        sample["path_rel"],
                "category_gt": sample["category_gt"],
                "label_gt":    sample["label_gt"],
                "prediction":  prediction,
                "correct":     prediction == sample["label_gt"],
            })

        except Exception as e:
            error_count += 1
            progress.write(f"ERROR: {Path(sample['path']).name}: {e}")
            results.append({
                "path":        sample.get("path_rel", ""),
                "category_gt": sample.get("category_gt", ""),
                "label_gt":    sample.get("label_gt", ""),
                "prediction":  "ERROR",
                "correct":     False,
            })

        progress.set_postfix({
            "REAL": real_count, "FAKE": fake_count,
            "UNK":  unknown_count, "ERR": error_count,
        })

    elapsed = time.time() - start_time
    print(f"\nTempo totale: {elapsed:.0f}s ({elapsed/len(samples):.1f}s/img)")
    return results


# ── Metriche ─────────────────────────────────────────────
def compute_metrics(results):
    CATEGORIES = ["original", "transfer", "redigital"]
    metrics = {}

    print(f"\n{'='*60}")
    print(f"METRICHE — Qwen-{args.model_size} Zero-Shot (senza esperti)")
    print(f"{'='*60}")

    for category in CATEGORIES + ["overall"]:
        subset = results if category == "overall" \
                 else [r for r in results if r["category_gt"] == category]
        valid  = [r for r in subset if r["prediction"] not in ["ERROR", "UNKNOWN"]]
        unk    = sum(1 for r in subset if r["prediction"] in ["UNKNOWN", "ERROR"])

        if not valid:
            continue

        tp = sum(1 for r in valid if r["prediction"] == "FAKE" and r["label_gt"] == "FAKE")
        fp = sum(1 for r in valid if r["prediction"] == "FAKE" and r["label_gt"] == "REAL")
        tn = sum(1 for r in valid if r["prediction"] == "REAL" and r["label_gt"] == "REAL")
        fn = sum(1 for r in valid if r["prediction"] == "REAL" and r["label_gt"] == "FAKE")

        acc  = (tp + tn) / len(valid)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"[{category.upper():10}] n={len(valid)} (unk={unk}) | "
              f"Acc={acc:.3f} | F1={f1:.3f} | "
              f"Prec={prec:.3f} | Rec={rec:.3f} | "
              f"FPR={fpr:.3f} | FNR={fnr:.3f}")

        metrics[category] = {
            "total": len(valid), "unknown": unk,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        }

    return metrics


# ── Salvataggio ──────────────────────────────────────────
def save_results(results, metrics):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

    # Salva JSON risultati prima delle metriche
    results_json = RESULTS_FILE.replace(".csv", ".json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    # Salva metriche
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metriche salvate in: {METRICS_FILE}")

    # Salva CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category_gt", "label_gt", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Risultati salvati in: {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGPU disponibili: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")

    print("\nCaricamento modello...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
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

    # Quick test
    print("\nQuick test...")
    test_messages = [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the single word: REAL"}
    ]}]
    test_chat   = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = processor(text=[test_chat], return_tensors="pt").to(model.device)
    with torch.no_grad():
        test_out = model.generate(**test_inputs, max_new_tokens=5, do_sample=False)
    test_answer = processor.decode(test_out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    print(f"Quick test answer: '{test_answer}'")
    if "REAL" not in test_answer.upper():
        print("WARNING: quick test non ha risposto REAL — potrebbe essere un problema multi-GPU.")
    else:
        print("OK: quick test superato.\n")

    # Pipeline
    samples = load_samples(args.input_json)
    results = run_inference(model, processor, samples)
    metrics = compute_metrics(results)
    save_results(results, metrics)
    print("\nCompletato!")