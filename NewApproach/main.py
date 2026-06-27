import os
import sys
import io

# Configure stdout and stderr to use UTF-8 to prevent UnicodeEncodeError in Windows consoles
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import time
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import configuration as config 
from original_model import EmbeddingSubstitutionModel
from t5_model import DynamicT5Simplifier
from byt5_model import CharacterTrieNeuralSimplifier
from MLM_model import MLMNeuralSimplifier
from evaluator import SimplificationEvaluator
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

# Global caching store for shared model/tokenizer instances
MODEL_CACHE = {}
TOKENIZER_CACHE = {}

def get_cached_model_and_tokenizer(model_id: str, base_model_name: str):
    cache_key = (model_id, base_model_name)
    if cache_key not in MODEL_CACHE:
        print(f"📥 Loading and caching {model_id} model weights ({base_model_name})...")
        TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(base_model_name)
        if model_id == "EMB_SUB":
            MODEL_CACHE[cache_key] = AutoModel.from_pretrained(base_model_name)
        elif model_id == "MLM":
            MODEL_CACHE[cache_key] = AutoModelForMaskedLM.from_pretrained(base_model_name)
        elif model_id in ["T5", "BYT5", "BYTAM"]:
            MODEL_CACHE[cache_key] = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        else:
            raise ValueError(f"Unknown model type for caching: {model_id}")
    return TOKENIZER_CACHE[cache_key], MODEL_CACHE[cache_key]

def initialize_model(model_id: str, raw_vocab_set: set, active_exempt_set: set, base_model_name: str):
    """Factory function handling architecture isolation mapping."""
    tokenizer, model = get_cached_model_and_tokenizer(model_id, base_model_name)
    if model_id == "EMB_SUB":
        return EmbeddingSubstitutionModel(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name,
            tokenizer=tokenizer, embedder=model
        )
    elif model_id == "MLM":
        return MLMNeuralSimplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name,
            tokenizer=tokenizer, model=model
        )
    elif model_id == "T5":
        return DynamicT5Simplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name,
            tokenizer=tokenizer, model=model
        )
    elif model_id in ["BYT5", "BYTAM"]:
        return CharacterTrieNeuralSimplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name,
            tokenizer=tokenizer, model=model
        )
    raise ValueError(f"Unknown model key: {model_id}")

def clean_and_detokenize(text_list: list) -> list:
    """Cleans spaces around punctuation marks to avoid metric penalties."""
    cleaned_list = []
    for text in text_list:
        cleaned = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
        cleaned = cleaned.replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
        cleaned = cleaned.replace(" ' ", "'").replace(" n't", "n't")
        cleaned_list.append(cleaned.strip())
    return cleaned_list

def compile_distribution_stats(metrics_dict: dict) -> dict:
    """Computes Avg, Median, Min, Max, and P90 distributions for raw metric lists."""
    expanded_stats = {}
    for metric_key, scores in metrics_dict.items():
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
            scores_p = np.array(scores)
            expanded_stats[f"{metric_key}_Avg"] = round(float(np.mean(scores_p)), 4)
            expanded_stats[f"{metric_key}_Median"] = round(float(np.median(scores_p)), 4)
            expanded_stats[f"{metric_key}_Min"] = round(float(np.min(scores_p)), 4)
            expanded_stats[f"{metric_key}_Max"] = round(float(np.max(scores_p)), 4)
            expanded_stats[f"{metric_key}_P90"] = round(float(np.percentile(scores_p, 90)), 4)
        else:
            expanded_stats[metric_key] = scores
    return expanded_stats

def write_final_csv_report(master_results: list, output_csv_file: str):
    """Generates the final output CSV file."""
    df_results = pd.DataFrame(master_results)
    df_results.to_csv(output_csv_file, index=False)

def save_comparison_samples(dataset_name: str, vocab_name: str, exempt_mode: str, model_id: str, source_sentences: list, transformed_sentences: list, num_samples: int = 10, seed: int = 42):
    """Appends comparison rows (original and transformed) for a given setting to the dataset's comparison CSV file."""
    import random
    import csv
    import os
    
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    n = min(len(source_sentences), len(transformed_sentences))
    if n == 0:
        return
        
    # Deterministically select the same indices for this dataset
    rng = random.Random(seed)
    sample_indices = rng.sample(range(n), min(num_samples, n))
    sample_indices.sort()
    
    filename = os.path.join(samples_dir, f"{dataset_name}_comparison.csv")
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Index", "Original_Sentence", "Vocabulary", "Exempt_Mode", "Model_ID", "Transformed_Sentence"])
            
        for idx in sample_indices:
            writer.writerow([
                idx,
                source_sentences[idx],
                vocab_name,
                exempt_mode,
                model_id,
                transformed_sentences[idx]
            ])

def run_evaluation_pipeline():
    active_datasets = [ds for ds, enabled in config.DATASETS.items() if enabled]
    active_vocab_paths = [path for path, enabled in config.VOCAB_FILES.items() if enabled]
    active_exempt_modes = config.ACTIVE_EXEMPT_MODES
    active_models = [m for m, enabled in config.MODEL_SELECTION.items() if enabled]
    
    # Pre-load dataset evaluators
    evaluators = {ds: SimplificationEvaluator(dataset_name=ds) for ds in active_datasets}
    for ev in evaluators.values():
        ev.load_data()

    total_evaluations = len(active_vocab_paths) * len(active_exempt_modes) * len(active_models) * len(active_datasets)
    completed_evaluations = 0

    print("=======================================================")
    print(f"🚀 OPTIMIZED CACHING MATRIX RUN STARTED: {total_evaluations} Total Metrics Points")
    print("=======================================================")
    
    master_results = []
    global_start_time = time.time()
    single_job_duration = None
    output_csv, output_csv_final = "MASTER_BENCHMARK_MATRIX.csv", "output.csv"
    
    for f in [output_csv, output_csv_final]:
        if os.path.exists(f): os.remove(f)

    # Clear samples directory
    import shutil
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)

    # 🔄 LOOP 1: VOCABULARIES
    for vc_idx, vocab_path in enumerate(active_vocab_paths, 1):
        if not os.path.exists(vocab_path):
            print(f"⚠️ Vocabulary file missing: {vocab_path}. Skipping.")
            completed_evaluations += len(active_exempt_modes) * len(active_models) * len(active_datasets)
            continue
            
        vocab_name = os.path.basename(vocab_path)
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            raw_vocab_set = {line.strip() for line in vf if line.strip()}

        # 🔄 LOOP 2: EXEMPT MODES
        for ex_idx, exempt_mode in enumerate(active_exempt_modes, 1):
            active_exempt_set = config.get_exempt_vocabulary(exempt_mode)

            # 🔄 LOOP 3: MODELS
            for md_idx, model_id in enumerate(active_models, 1):
                print(f"\n📁 Vocab: {vocab_name} ({vc_idx}/{len(active_vocab_paths)}) | 🛡️ Exempt: {exempt_mode} ({ex_idx}/{len(active_exempt_modes)}) | 🤖 Model: {model_id} ({md_idx}/{len(active_models)})")

                try:
                    model_instance = initialize_model(model_id, raw_vocab_set, active_exempt_set, config.MODEL_NAMES.get(model_id))
                except Exception as e:
                    print(f"❌ Initialization runtime failure on [{model_id}]: {e}")
                    completed_evaluations += len(active_datasets)
                    continue

                # Run inference caching pool for Turk/Asset
                wiki_source = evaluators["turk_corpus"].source_sentences if "turk_corpus" in evaluators else (evaluators["asset"].source_sentences if "asset" in evaluators else None)
                wiki_predictions_cache = None
                
                if wiki_source:
                    job_start_clock = time.time()
                    try:
                        print(f"⚡ Running Inference ONCE for shared Wiki359 Pool ({model_id})...")
                        wiki_predictions_cache = clean_and_detokenize(model_instance.batch_simplify(wiki_source))
                    except Exception as e:
                        print(f"❌ Inference processing crash: {e}")
                    job_duration = time.time() - job_start_clock

                # Evaluate active datasets from pool cache
                for target_ds in ["turk_corpus", "asset"]:
                    if target_ds in evaluators and wiki_predictions_cache:
                        print(f"📊 Evaluating {target_ds} from inference cache...")
                        try:
                            metrics = compile_distribution_stats(evaluators[target_ds].compute_metrics(wiki_predictions_cache))
                            run_summary = {"Dataset": target_ds, "Vocabulary": vocab_name, "Exempt_Mode": exempt_mode, "Model_ID": model_id, **metrics}
                            master_results.append(run_summary)
                            pd.DataFrame([run_summary]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                            save_comparison_samples(
                                dataset_name=target_ds,
                                vocab_name=vocab_name,
                                exempt_mode=exempt_mode,
                                model_id=model_id,
                                source_sentences=evaluators[target_ds].source_sentences,
                                transformed_sentences=wiki_predictions_cache
                            )
                        except Exception as e:
                            print(f"❌ Metrics evaluation crash on {target_ds}: {e}")
                        completed_evaluations += 1

                # Evaluate independent SICK dataset pool
                if "sick" in evaluators:
                    job_start_clock = time.time()
                    try:
                        print(f"⚡ Running Inference for independent SICK Pool ({model_id})...")
                        sick_preds = clean_and_detokenize(model_instance.batch_simplify(evaluators["sick"].source_sentences))
                        metrics = compile_distribution_stats(evaluators["sick"].compute_metrics(sick_preds))
                        run_summary = {"Dataset": "sick", "Vocabulary": vocab_name, "Exempt_Mode": exempt_mode, "Model_ID": model_id, **metrics}
                        master_results.append(run_summary)
                        pd.DataFrame([run_summary]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                        save_comparison_samples(
                            dataset_name="sick",
                            vocab_name=vocab_name,
                            exempt_mode=exempt_mode,
                            model_id=model_id,
                            source_sentences=evaluators["sick"].source_sentences,
                            transformed_sentences=sick_preds
                        )
                    except Exception as e:
                        print(f"❌ Processing crash on SICK: {e}")
                    job_duration = time.time() - job_start_clock
                    completed_evaluations += 1

                # Dynamic runtime estimation tracking
                if single_job_duration is None: single_job_duration = job_duration
                remaining_jobs = total_evaluations - completed_evaluations
                if remaining_jobs > 0:
                    est_remaining_mins = (remaining_jobs * (single_job_duration / len(active_datasets))) / 60.0
                    print(f"⏳ Estimated Matrix Time Remaining: {est_remaining_mins:.1f} minutes")

                # Free GPU memory to prevent OOM across configurations
                del model_instance
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_elapsed_mins = (time.time() - global_start_time) / 60.0
    if master_results:
        write_final_csv_report(master_results, output_csv_final)

    print(f"\n💾 Summary written to: {output_csv_final}\n📊 Backend CSV compiled: {output_csv}\n⏱️ Total wall time: {total_elapsed_mins:.2f} minutes")

if __name__ == "__main__":
    run_evaluation_pipeline()