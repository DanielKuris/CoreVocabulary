import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import configuration as config 
from original_model import EmbeddingSubstitutionModel
from t5_model import DynamicT5Simplifier
from byt5_model import CharacterTrieNeuralSimplifier
from MLM_model import MLMNeuralSimplifier
from evaluator import SimplificationEvaluator

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
            expanded_stats[f"{metric_key}_Avg"] = float(np.mean(scores_p))
            expanded_stats[f"{metric_key}_Median"] = float(np.median(scores_p))
            expanded_stats[f"{metric_key}_Min"] = float(np.min(scores_p))
            expanded_stats[f"{metric_key}_Max"] = float(np.max(scores_p))
            expanded_stats[f"{metric_key}_P90"] = float(np.percentile(scores_p, 90))
            
            expanded_stats[metric_key] = expanded_stats[f"{metric_key}_Avg"]
        else:
            expanded_stats[metric_key] = scores
    return expanded_stats

def initialize_model(model_id: str, raw_vocab_set: set, active_exempt_set: set, base_model_name: str):
    """Factory function handling architecture isolation mapping."""
    if model_id == "EMB_SUB":
        return EmbeddingSubstitutionModel(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name
        )
    elif model_id == "MLM":
        return MLMNeuralSimplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name
        )
    elif model_id == "T5":
        return DynamicT5Simplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name
        )
    elif model_id in ["BYT5", "BYTAM"]:
        return CharacterTrieNeuralSimplifier(
            vocabulary=raw_vocab_set, exempt_vocabulary=active_exempt_set, model_name=base_model_name
        )
    raise ValueError(f"Unknown model key: {model_id}")

def write_scannable_dashboard(master_results: list, active_datasets: list, total_vocabs: int, elapsed_mins: float, output_txt_file: str):
    """Generates the human-scannable dashboard text file."""
    df_results = pd.DataFrame(master_results)
    with open(output_txt_file, "w", encoding="utf-8") as out_file:
        out_file.write("=======================================================\n")
        out_file.write("📊 ProjectODR Highly-Optimized Caching Evaluation Dashboard Matrix\n")
        out_file.write("=======================================================\n")
        out_file.write(f"⏱️ Total Matrix Runtime Duration: {elapsed_mins:.2f} minutes\n")
        out_file.write(f"📦 Source Datasets Evaluated   : {', '.join(active_datasets)}\n")
        out_file.write(f"📝 Target Constraints Managed  : {total_vocabs} configuration files\n\n")
        out_file.write(df_results.to_string(index=False))
        out_file.write("\n\n=== END OF AUTOMATED MATRIC EVALUATION DATA REPORT ===")

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
    output_csv, output_txt = "MASTER_BENCHMARK_MATRIX.csv", "output.txt"
    
    for f in [output_csv, output_txt]:
        if os.path.exists(f): os.remove(f)

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

    total_elapsed_mins = (time.time() - global_start_time) / 60.0
    if master_results:
        write_scannable_dashboard(master_results, active_datasets, len(active_vocab_paths), total_elapsed_mins, output_txt)

    print(f"\n💾 Summary written to: {output_txt}\n📊 Backend CSV compiled: {output_csv}\n⏱️ Total wall time: {total_elapsed_mins:.2f} minutes")

if __name__ == "__main__":
    run_evaluation_pipeline()