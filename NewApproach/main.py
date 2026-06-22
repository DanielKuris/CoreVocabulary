import os
import sys
import time
import warnings

# Mute Hugging Face system warnings and caching alerts
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

from src.evaluator import SimplificationEvaluator
from src.neural_model import MLMNeuralSimplifier
from src.seq2seq_model import Seq2SeqNeuralSimplifier
from src.by_t5_model import CharacterTrieNeuralSimplifier
import src.configuration as config

def load_vocabulary_from_file(file_path: str) -> set:
    """Parses a target vocabulary file and strips punctuation formatting."""
    if not os.path.exists(file_path):
        print(f"❌ Error: The file '{file_path}' was not found on drive.")
        sys.exit(1)
        
    vocab_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                clean_token = token.strip(".,!?;:()\"'").lower()
                if clean_token:
                    vocab_set.add(clean_token)
    return vocab_set

def run_model_inference(model_key: str, simplifier_class, model_name: str, target_vocab: set, evaluator) -> tuple:
    """Helper function to cleanly handle model setup, execution timing, and evaluation."""
    metrics = {"SARI": 0.0, "BLEU": 0.0, "METEOR": 0.0, "BERTScore": 0.0}
    duration = 0.0
    
    if config.MODEL_SELECTION[model_key]:
        simplifier = simplifier_class(
            vocabulary=target_vocab,
            exempt_vocabulary=config.EXEMPT_VOCABULARY,
            model_name=model_name
        )
        t0 = time.time()
        predictions = simplifier.batch_simplify(evaluator.source_sentences)
        duration = time.time() - t0
        metrics = evaluator.compute_metrics(predictions)
        
        # Squelch disabled metric metrics
        for metric, enabled in config.METRIC_SELECTION.items():
            if not enabled:
                metrics[metric] = 0.0
                
    return metrics, duration

def print_master_dashboard(all_results: list):
    """Generates a clean comparative terminal summary matrix grid reporting metrics rows."""
    print("\n" + "="*53 + " MASTER BENCHMARK MATRIX " + "="*54)
    print("-" * 132)
    print(f"| {'Dataset':<12} | {'Vocabulary':<24} | {'Model Type':<14} | {'SARI':<8} | {'BLEU':<8} | {'METEOR':<8} | {'BERTScore':<10} | {'Time':<8} |")
    print(f"|{'-'*14}|{'-'*26}|{'-'*16}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*10}|")
    
    for res in all_results:
        # Determine if this row is the absolute first model printed for this configuration
        is_first_printed_row = True
        
        if config.MODEL_SELECTION["MLM"]:
            d_label = res['dataset'] if is_first_printed_row else ""
            v_label = res['filename'] if is_first_printed_row else ""
            print(f"| {d_label:<12} | {v_label:<24} | DistilBERT MLM | {res['mlm_sari']:<8.2f} | {res['mlm_bleu']:<8.2f} | {res['mlm_meteor']:<8.2f} | {res['mlm_bert']:<10.2f} | {res['mlm_time']:>5.1f}s   |")
            is_first_printed_row = False
            
        if config.MODEL_SELECTION["T5"]:
            d_label = res['dataset'] if is_first_printed_row else ""
            v_label = res['filename'] if is_first_printed_row else ""
            print(f"| {d_label:<12} | {v_label:<24} | T5-Small S2S   | {res['t5_sari']:<8.2f} | {res['t5_bleu']:<8.2f} | {res['t5_meteor']:<8.2f} | {res['t5_bert']:<10.2f} | {res['t5_time']:>5.1f}s   |")
            is_first_printed_row = False
            
        if config.MODEL_SELECTION["BYT5"]:
            d_label = res['dataset'] if is_first_printed_row else ""
            v_label = res['filename'] if is_first_printed_row else ""
            print(f"| {d_label:<12} | {v_label:<24} | ByT5 Positive  | {res['byt5_sari']:<8.2f} | {res['byt5_bleu']:<8.2f} | {res['byt5_meteor']:<8.2f} | {res['byt5_bert']:<10.2f} | {res['byt5_time']:>5.1f}s   |")
            is_first_printed_row = False
            
        print("-" * 132)
    print("=" * 132 + "\n")

def run_comprehensive_neural_comparison():
    print("==============================================================")
    print("🚀 Running Phase 2: Multi-Architectural Empirical Evaluation")
    print("==============================================================\n")

    all_results = []

    for dataset_name, dataset_enabled in config.DATASETS.items():
        if not dataset_enabled:
            print(f"⏭️ Skipping Dataset configuration: {dataset_name.upper()}")
            continue
            
        print(f"\n🧱 Initializing Test Bench Dataset: {dataset_name.upper()} 🧱")
        evaluator = SimplificationEvaluator(dataset_name=dataset_name)
        evaluator.load_data()

        for vocab_file, vocab_enabled in config.VOCAB_FILES.items():
            if not vocab_enabled:
                print(f"   ⏭️ Skipping Lexicon configuration: {vocab_file}")
                continue
                
            print(f"\n   📊 Processing Target Config: {vocab_file}")
            target_vocabulary = load_vocabulary_from_file(vocab_file)
            vocab_size = len(target_vocabulary)
            
            # Isolate just the filename for matrix display cleanliness
            display_name = os.path.basename(vocab_file)

            # Parallel inference tracking blocks cleanly handled via abstract helpers
            mlm_m, mlm_d = run_model_inference("MLM", MLMNeuralSimplifier, "distilbert-base-uncased", target_vocabulary, evaluator)
            t5_m, t5_d   = run_model_inference("T5", Seq2SeqNeuralSimplifier, "t5-small", target_vocabulary, evaluator)
            byt5_m, byt5_d = run_model_inference("BYT5", CharacterTrieNeuralSimplifier, "google/byt5-small", target_vocabulary, evaluator)

            all_results.append({
                "dataset": dataset_name.upper(), "filename": display_name, "vocab_size": vocab_size,
                "mlm_sari": mlm_m['SARI'], "mlm_bleu": mlm_m['BLEU'], "mlm_meteor": mlm_m['METEOR'], "mlm_bert": mlm_m['BERTScore'], "mlm_time": mlm_d,
                "t5_sari": t5_m['SARI'], "t5_bleu": t5_m['BLEU'], "t5_meteor": t5_m['METEOR'], "t5_bert": t5_m['BERTScore'], "t5_time": t5_d,
                "byt5_sari": byt5_m['SARI'], "byt5_bleu": byt5_m['BLEU'], "byt5_meteor": byt5_m['METEOR'], "byt5_bert": byt5_m['BERTScore'], "byt5_time": byt5_d
            })

    print_master_dashboard(all_results)

if __name__ == "__main__":
    run_comprehensive_neural_comparison()