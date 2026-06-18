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
parser.add_argument("--input_json", type=str, default="/work/project/pipeline/test_with_experts.json",
                    help="JSON con i risultati degli esperti da pipeline_merge.py")
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

MODEL_PATH   = MODEL_MAP[args.model_size]
METRICS_FILE = f"/work/project/metrics_final_{args.model_size}.json"
RESULTS_FILE = f"/work/project/results_final_{args.model_size}.csv"
DATASET_ROOT = "/mnt/ssd1/teglia/rrdataset/RRDataset_final"

print(f"Usando modello: {MODEL_PATH}")
print(f"Input JSON:     {args.input_json}")


# ── Carica campioni dal JSON finale ──────────────────────
def load_samples(input_json):
    with open(input_json) as f:
        samples = json.load(f)
    print(f"Totale immagini: {len(samples)}")
    return samples


# ── Inferenza ────────────────────────────────────────────
def run_inference(model, processor, samples):
    results = []
    real_count = fake_count = unknown_count = error_count = 0
    start_time = time.time()

    progress = tqdm(samples, desc="Inference", dynamic_ncols=True)

    for sample in progress:
        try:
            # Usa path_server per caricare l'immagine
            img_path = sample.get("path_server", os.path.join(DATASET_ROOT, sample["path"]))
            img = Image.open(img_path).convert("RGB")

            # Usa il prompt già costruito nel JSON
            # Contiene PROMPT_STRUCTURED + risposte esperti con pesi
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  sample["prompt"]},
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
                progress.write(f"UNKNOWN: {Path(img_path).name}: '{answer}'")

            results.append({
                "path":         sample["path"],
                "category_gt":  sample["category_gt"],
                "label_gt":     sample["label_gt"],
                "prediction":   prediction,
                "correct":      prediction == sample["label_gt"],
                # Risposte esperti per analisi
                "expert_original":  sample["experts"]["original"],
                "expert_transfer":  sample["experts"]["transfer"],
                "expert_redigital": sample["experts"]["redigital"],
                "detector_category": sample["detector"]["predicted_category"],
                "w_original":  sample["detector"]["weights"]["original"],
                "w_transfer":  sample["detector"]["weights"]["transfer"],
                "w_redigital": sample["detector"]["weights"]["redigital"],
            })

        except Exception as e:
            error_count += 1
            progress.write(f"ERROR: {e}")
            results.append({
                "path":        sample.get("path", ""),
                "category_gt": sample.get("category_gt", ""),
                "label_gt":    sample.get("label_gt", ""),
                "prediction":  "ERROR",
                "correct":     False,
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
    CATEGORIES = ["original", "transfer", "redigital"]
    metrics = {}

    for category in CATEGORIES + ["overall"]:
        subset = results if category == "overall" \
                 else [r for r in results if r["category_gt"] == category]
        subset = [r for r in subset if r["prediction"] not in ["ERROR", "UNKNOWN"]]
        total = len(subset)
        if not subset:
            continue

        tp = sum(1 for r in subset if r["prediction"] == "FAKE" and r["label_gt"] == "FAKE")
        fp = sum(1 for r in subset if r["prediction"] == "FAKE" and r["label_gt"] == "REAL")
        tn = sum(1 for r in subset if r["prediction"] == "REAL" and r["label_gt"] == "REAL")
        fn = sum(1 for r in subset if r["prediction"] == "REAL" and r["label_gt"] == "FAKE")

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

    fieldnames = ["path", "category_gt", "label_gt", "prediction", "correct",
                  "expert_original", "expert_transfer", "expert_redigital",
                  "detector_category", "w_original", "w_transfer", "w_redigital"]
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    if hasattr(model, "hf_device_map"):
        devices_used = sorted(set(str(v) for v in model.hf_device_map.values()))
        print(f"Device usati: {devices_used}")
    for g in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(g) / 1024**3
        print(f"VRAM GPU {g}: {allocated:.2f}GB")

    # ── Quick test ───────────────────────────────────────
    print("\nQuick test...")
    test_messages = [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the single word: REAL"}
    ]}]
    test_chat   = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = processor(text=[test_chat], return_tensors="pt").to(model.device)
    with torch.no_grad():
        test_output = model.generate(**test_inputs, max_new_tokens=5, do_sample=False)
    test_text   = processor.decode(test_output[0], skip_special_tokens=True)
    test_answer = test_text.split("assistant")[-1].strip()
    print(f"Quick test answer: '{test_answer}'")
    if "REAL" not in test_answer.upper():
        print("WARNING: quick test fallito.")
        raise SystemExit(1)
    print("OK: quick test superato.\n")

    # ── Pipeline principale ──────────────────────────────
    samples = load_samples(args.input_json)
    results = run_inference(model, processor, samples)
    metrics = compute_metrics(results)
    save_results(results, metrics)
    print("\nCompletato!")