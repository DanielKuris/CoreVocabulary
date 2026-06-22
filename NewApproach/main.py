import os
import sys
import time
import pandas as pd

# Fix path resolution for scripts located inside the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Dynamic imports from the 'src' folder layout
import configuration as config 
from original_model import EmbeddingSubstitutionModel
from seq2seq_model import Seq2SeqNeuralSimplifier 
from neural_model import MLMNeuralSimplifier
from evaluator import SimplificationEvaluator

def clean_and_detokenize(text_list: list) -> list:
    """
    Cleans structural sub-token spaces separating punctuation marks
    to prevent score penalties in standard metric engines.
    """
    cleaned_list = []
    for text in text_list:
        cleaned = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
        cleaned = cleaned.replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
        cleaned = cleaned.replace(" ' ", "'").replace(" n't", "n't")
        cleaned_list.append(cleaned.strip())
    return cleaned_list

def run_evaluation_pipeline():
    # -----------------------------------------------------------------
    # 📊 PRE-COMPUTE SETUP & TARGET MATRIX BOUNDARIES
    # -----------------------------------------------------------------
    active_datasets = [ds for ds, enabled in config.DATASETS.items() if enabled]
    active_vocab_paths = [path for path, enabled in config.VOCAB_FILES.items() if enabled]
    active_exempt_modes = config.ACTIVE_EXEMPT_MODES
    active_models = [m for m, enabled in config.MODEL_SELECTION.items() if enabled]
    
    total_vocabs = len(active_vocab_paths)
    total_exempt = len(active_exempt_modes)
    total_models = len(active_models)
    
    # Pre-load all required evaluators into a static dictionary at startup
    evaluators = {}
    for dataset_name in active_datasets:
        ev = SimplificationEvaluator(dataset_name=dataset_name)
        ev.load_data()
        evaluators[dataset_name] = ev

    # Calculate true total unique evaluation jobs to keep tracking accurate
    total_evaluations = total_vocabs * total_exempt * total_models * len(active_datasets)
    completed_evaluations = 0

    print("=======================================================")
    print(f"🚀 OPTIMIZED CACHING MATRIX RUN STARTED: {total_evaluations} Total Metrics Points")
    print("=======================================================")
    
    master_results = []
    global_start_time = time.time()
    single_job_duration = None
    
    output_csv_file = "MASTER_BENCHMARK_MATRIX.csv"
    output_txt_file = "output.txt"
    
    # Startup Cache Reset Guardrails
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    if os.path.exists(output_txt_file):
        os.remove(output_txt_file)

    # 🔄 LOOP 1: VOCABULARIES
    for vc_idx, vocab_path in enumerate(active_vocab_paths, 1):
        if not os.path.exists(vocab_path):
            print(f"⚠️ Vocabulary file missing: {vocab_path}. Skipping dynamic configurations.")
            completed_evaluations += total_exempt * total_models * len(active_datasets)
            continue
            
        vocab_name = os.path.basename(vocab_path)
        with open(vocab_path, 'r', encoding='utf-8') as vf:
            raw_vocab_set = {line.strip() for line in vf if line.strip()}

        # 🔄 LOOP 2: EXEMPT MODES
        for ex_idx, exempt_mode in enumerate(active_exempt_modes, 1):
            active_exempt_set = config.get_exempt_vocabulary(exempt_mode)

            # 🔄 LOOP 3: MODELS
            for md_idx, model_id in enumerate(active_models, 1):
                
                print("\n" + "=" * 40)
                print(f"📁 Vocab: {vocab_name} ({vc_idx}/{total_vocabs})")
                print(f"🛡️ Exempt Mode: {exempt_mode} ({ex_idx}/{total_exempt})")
                print(f"🤖 Computing Model: {model_id} ({md_idx}/{total_models})")
                print("=" * 40)

                base_model_name = config.MODEL_NAMES.get(model_id)

                # Initialize Model Architecture Instance
                try:
                    if model_id == "EMB_SUB":
                        model_instance = EmbeddingSubstitutionModel(
                            vocabulary=raw_vocab_set,
                            exempt_vocabulary=active_exempt_set,
                            model_name=base_model_name
                        )
                    elif model_id == "MLM":
                        model_instance = MLMNeuralSimplifier(
                            vocabulary=raw_vocab_set,
                            exempt_vocabulary=active_exempt_set,
                            model_name=base_model_name
                        )
                    elif model_id in ["T5", "BYTAM", "BYT5"]:
                        model_instance = Seq2SeqNeuralSimplifier(
                            vocabulary=raw_vocab_set,
                            exempt_vocabulary=active_exempt_set,
                            model_name=base_model_name
                        )
                    else:
                        print(f"⚠️ Unknown model key in config: {model_id}")
                        completed_evaluations += len(active_datasets)
                        continue
                except Exception as e:
                    print(f"❌ Initialization runtime failure on [{model_id}]: {e}")
                    completed_evaluations += len(active_datasets)
                    continue

                # -------------------------------------------------------------
                # 🏃 BLOCK A: WIKI359 POOL GENERATION CACHE (For Turk & Asset)
                # -------------------------------------------------------------
                wiki_predictions_cache = None
                wiki_source_sentences = None
                
                # Extract the sentences from whoever is active first to establish the baseline pool
                if "turk_corpus" in evaluators:
                    wiki_source_sentences = evaluators["turk_corpus"].source_sentences
                elif "asset" in evaluators:
                    wiki_source_sentences = evaluators["asset"].source_sentences

                if wiki_source_sentences:
                    job_start_clock = time.time()
                    try:
                        print(f"⚡ Running Inference ONCE for shared Wiki359 Pool ({model_id})...")
                        raw_pred = model_instance.batch_simplify(wiki_source_sentences)
                        wiki_predictions_cache = clean_and_detokenize(raw_pred)
                    except Exception as e:
                        print(f"❌ Inference processing crash on Wiki359 shared loop: {e}")
                    job_duration = time.time() - job_start_clock

                # 📊 Evaluate Turk Corpus using the Cached Predictions
                if "turk_corpus" in evaluators:
                    print("📊 Evaluating Turk Corpus from active inference cache...")
                    try:
                        if wiki_predictions_cache:
                            metrics = evaluators["turk_corpus"].compute_metrics(wiki_predictions_cache)
                            run_summary = {"Dataset": "turk_corpus", "Vocabulary": vocab_name, "Exempt_Mode": exempt_mode, "Model_ID": model_id, **metrics}
                            master_results.append(run_summary)
                            pd.DataFrame([run_summary]).to_csv(output_csv_file, mode='a', header=not os.path.exists(output_csv_file), index=False)
                    except Exception as e:
                        print(f"❌ Metrics evaluation crash on Turk Corpus: {e}")
                    completed_evaluations += 1

                # 📊 Evaluate ASSET using the EXACT SAME Cached Predictions
                if "asset" in evaluators:
                    print("📊 Evaluating ASSET from active inference cache (INFERENCE SKIPPED! ✅)...")
                    try:
                        if wiki_predictions_cache:
                            metrics = evaluators["asset"].compute_metrics(wiki_predictions_cache)
                            run_summary = {"Dataset": "asset", "Vocabulary": vocab_name, "Exempt_Mode": exempt_mode, "Model_ID": model_id, **metrics}
                            master_results.append(run_summary)
                            pd.DataFrame([run_summary]).to_csv(output_csv_file, mode='a', header=not os.path.exists(output_csv_file), index=False)
                    except Exception as e:
                        print(f"❌ Metrics evaluation crash on ASSET: {e}")
                    completed_evaluations += 1

                # -------------------------------------------------------------
                # 🏃 BLOCK B: SICK POOL GENERATION (Distinct 400 Sentences)
                # -------------------------------------------------------------
                if "sick" in evaluators:
                    job_start_clock = time.time()
                    try:
                        print(f"⚡ Running Inference for independent SICK Pool ({model_id})...")
                        raw_pred = model_instance.batch_simplify(evaluators["sick"].source_sentences)
                        sick_preds = clean_and_detokenize(raw_pred)
                        
                        metrics = evaluators["sick"].compute_metrics(sick_preds)
                        run_summary = {"Dataset": "sick", "Vocabulary": vocab_name, "Exempt_Mode": exempt_mode, "Model_ID": model_id, **metrics}
                        master_results.append(run_summary)
                        pd.DataFrame([run_summary]).to_csv(output_csv_file, mode='a', header=not os.path.exists(output_csv_file), index=False)
                    except Exception as e:
                        print(f"❌ Processing crash on SICK evaluation tier: {e}")
                    
                    job_duration = time.time() - job_start_clock
                    completed_evaluations += 1

                # 🕒 ROLLING REGRESSION TIME CALCULATOR
                if single_job_duration is None:
                    single_job_duration = job_duration # Establish benchmark baseline
                    
                remaining_jobs = total_evaluations - completed_evaluations
                if remaining_jobs > 0:
                    # Divisor matches weight distribution now that datasets are completely symmetrical
                    est_remaining_mins = (remaining_jobs * (single_job_duration / len(active_datasets))) / 60.0
                    print(f"⏳ Estimated Matrix Time Remaining: {est_remaining_mins:.1f} minutes")

    # =====================================================================
    # 💾 DATA CONSOLIDATION & HUMAN SCANNABLE TEXT EXPORT
    # =====================================================================
    total_elapsed_mins = (time.time() - global_start_time) / 60.0

    if master_results:
        df_results = pd.DataFrame(master_results)
        with open(output_txt_file, "w", encoding="utf-8") as out_file:
            out_file.write("=======================================================\n")
            out_file.write("📊 ProjectODR Highly-Optimized Caching Evaluation Dashboard Matrix\n")
            out_file.write("=======================================================\n")
            out_file.write(f"⏱️ Total Matrix Runtime Duration: {total_elapsed_mins:.2f} minutes\n")
            out_file.write(f"📦 Source Datasets Evaluated   : {', '.join(active_datasets)}\n")
            out_file.write(f"📝 Target Constraints Managed  : {len(active_vocab_paths)} configuration files\n\n")
            out_file.write(df_results.to_string(index=False))
            out_file.write("\n\n=== END OF AUTOMATED MATRIC EVALUATION DATA REPORT ===")

        print("\n=======================================================")
        print(f"🎉 OPTIMIZED CACHING PIPELINE COMPLETE!")
        print(f"💾 Human-scannable metrics summary saved to: {output_txt_file}")
        print(f"📊 Tabular spreadsheet backend CSV written to: {output_csv_file}")
        print(f"⏱️ Total processing wall-clock time: {total_elapsed_mins:.2f} minutes")
        print("=======================================================")

if __name__ == "__main__":
    run_evaluation_pipeline()