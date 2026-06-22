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
        # Standardize space configurations around common punctuation
        cleaned = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
        cleaned = cleaned.replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
        # Handle contractions or apostrophe separations if they occur
        cleaned = cleaned.replace(" ' ", "'").replace(" n't", "n't")
        cleaned_list.append(cleaned.strip())
    return cleaned_list

def run_evaluation_pipeline():
    # -----------------------------------------------------------------
    # 📊 PRE-COMPUTE JOB MATRIX BOUNDARIES FROM CONFIG
    # -----------------------------------------------------------------
    active_datasets = [ds for ds, enabled in config.DATASETS.items() if enabled]
    active_vocab_paths = [path for path, enabled in config.VOCAB_FILES.items() if enabled]
    active_exempt_modes = config.ACTIVE_EXEMPT_MODES
    active_models = [m for m, enabled in config.MODEL_SELECTION.items() if enabled]
    
    total_datasets = len(active_datasets)
    total_vocabs = len(active_vocab_paths)
    total_exempt = len(active_exempt_modes)
    total_models = len(active_models)
    
    total_evaluations = total_datasets * total_vocabs * total_exempt * total_models
    completed_evaluations = 0

    print("=======================================================")
    print(f"🚀 MASTER MATRIX RUN STARTED: {total_evaluations} Total Job Assignments")
    print("=======================================================")
    
    master_results = []
    global_start_time = time.time()
    single_job_duration = None
    
    output_csv_file = "MASTER_BENCHMARK_MATRIX.csv"
    output_txt_file = "output.txt"
    
    # 🚨 PRECAUTIONARY STARTUP RESET
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    if os.path.exists(output_txt_file):
        os.remove(output_txt_file)

    # 🔄 LOOP 1: DATASETS
    for ds_idx, dataset_name in enumerate(active_datasets, 1):
        evaluator = SimplificationEvaluator(dataset_name=dataset_name)
        try:
            evaluator.load_data()
        except Exception as e:
            print(f"❌ Failed to load dataset {dataset_name}: {e}. Skipping data tier.")
            completed_evaluations += total_vocabs * total_exempt * total_models
            continue

        # 🔄 LOOP 2: VOCABULARIES
        for vc_idx, vocab_path in enumerate(active_vocab_paths, 1):
            if not os.path.exists(vocab_path):
                print(f"⚠️ Vocabulary file missing: {vocab_path}. Skipping row mapping.")
                completed_evaluations += total_exempt * total_models
                continue
                
            vocab_name = os.path.basename(vocab_path)
            with open(vocab_path, 'r', encoding='utf-8') as vf:
                raw_vocab_set = {line.strip() for line in vf if line.strip()}

            # 🔄 LOOP 3: EXEMPT MODES
            for ex_idx, exempt_mode in enumerate(active_exempt_modes, 1):
                active_exempt_set = config.get_exempt_vocabulary(exempt_mode)

                # Initialize runtime container for active models
                models_to_evaluate = {}
                
                # 🔄 LOOP 4: DYNAMIC MODELS INITIALIZATION
                for md_idx, model_id in enumerate(active_models, 1):
                    
                    # 🖨️ PROGRESS PRINT CHANNELS (Matches requested layout exactly)
                    print("\n" + "-" * 30)
                    print("Current progress:")
                    print(f"datasets {ds_idx}/{total_datasets}")
                    print(f"vocabularies {vc_idx}/{total_vocabs}")
                    print(f"exempt mode {ex_idx}/{total_exempt}")
                    print(f"computing model {md_idx}/{total_models} ({model_id})")
                    print("-" * 30)

                    base_model_name = config.MODEL_NAMES.get(model_id)

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
                            print(f"⚠️ Unknown model key in configuration selection: {model_id}")
                            completed_evaluations += 1
                            continue
                            
                    except Exception as e:
                        print(f"❌ Initialization runtime failure on [{model_id}]: {e}")
                        completed_evaluations += 1
                        continue

                    # 🏃 Run Inference & Matrix Benchmarking
                    job_start_clock = time.time()
                    try:
                        raw_predictions = model_instance.batch_simplify(evaluator.source_sentences)
                        
                        # 🩹 OPTIMIZATION: Detokenize split punctuation spacing before computing metrics
                        predictions = clean_and_detokenize(raw_predictions)
                        
                        metrics_report = evaluator.compute_metrics(predictions)
                        
                        # Apply boolean masking config filters
                        filtered_report = {
                            k: v for k, v in metrics_report.items() 
                            if config.METRIC_SELECTION.get(k.upper(), True)
                        }

                        run_summary = {
                            "Dataset": dataset_name,
                            "Vocabulary": vocab_name,
                            "Exempt_Mode": exempt_mode,
                            "Model_ID": model_id,
                            **filtered_report
                        }
                        master_results.append(run_summary)
                        
                        # 🛡️ LIVE DISK CHECKPOINT: Instant write protection per run iteration
                        df_checkpoint = pd.DataFrame([run_summary])
                        if not os.path.exists(output_csv_file):
                            df_checkpoint.to_csv(output_csv_file, index=False)
                        else:
                            df_checkpoint.to_csv(output_csv_file, mode='a', header=False, index=False)
                        
                    except Exception as e:
                        print(f"❌ Critical structural run failure during inference loop on {model_id}: {e}")
                    
                    # 🕒 TIME CALCULATION ENGINE
                    completed_evaluations += 1
                    job_end_clock = time.time()
                    
                    if single_job_duration is None:
                        single_job_duration = job_end_clock - job_start_clock
                        
                    remaining_jobs = total_evaluations - completed_evaluations
                    
                    if remaining_jobs > 0 and single_job_duration is not None:
                        est_remaining_mins = (remaining_jobs * single_job_duration) / 60.0
                        print(f"⏳ Estimated time remaining for matrix: {est_remaining_mins:.1f} minutes")
                    elif remaining_jobs == 0:
                        print("⏳ Final evaluation matrix assignment completed.")

    # =====================================================================
    # 💾 DATA CONSOLIDATION & HUMAN SCANNABLE TEXT EXPORT
    # =====================================================================
    total_elapsed_mins = (time.time() - global_start_time) / 60.0

    if master_results:
        df_results = pd.DataFrame(master_results)
        
        with open(output_txt_file, "w", encoding="utf-8") as out_file:
            out_file.write("=======================================================\n")
            out_file.write("📊 ProjectODR Master Model Evaluation Dashboard Matrix Report\n")
            out_file.write("=======================================================\n")
            out_file.write(f"⏱️ Total Matrix Runtime Duration: {total_elapsed_mins:.2f} minutes\n")
            out_file.write(f"📦 Source Datasets Evaluated   : {', '.join(active_datasets)}\n")
            out_file.write(f"📝 Target Constraints Managed  : {len(active_vocab_paths)} configuration files\n\n")
            out_file.write(df_results.to_string(index=False))
            out_file.write("\n\n=== END OF AUTOMATED MATRIC EVALUATION DATA REPORT ===")

        print("\n=======================================================")
        print(f"🎉 MASTER MATRIX PIPELINE COMPLETE!")
        print(f"💾 Human-scannable metrics summary saved to: {output_txt_file}")
        print(f"📊 Tabular spreadsheet backend CSV written to: {output_csv_file}")
        print(f"⏱️ Total processing wall-clock time: {total_elapsed_mins:.2f} minutes")
        print("=======================================================")
    else:
        print("\n❌ Evaluation batch loop processed 0 matrix entries. Output tracking reports aborted.")

if __name__ == "__main__":
    run_evaluation_pipeline()