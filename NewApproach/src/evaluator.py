import numpy as np
from datasets import load_dataset
import sacrebleu
import evaluate

class SimplificationEvaluator:
    def __init__(self, dataset_name: str = "turk_corpus"):
        self.dataset_name = dataset_name
        self.source_sentences = []
        self.references = []  # List of lists [num_refs, num_samples]
        
        # Initialize Hugging Face evaluation modules
        print("📥 Initializing METEOR and BERTScore evaluation layers...")
        self.meteor_metric = evaluate.load("meteor")
        self.bertscore_metric = evaluate.load("bertscore")

    def load_data(self):
        """Fetches and formats benchmarks automatically from Hugging Face."""
        print(f"📦 Loading '{self.dataset_name}' from GEM/wiki_auto_asset_turk via Hugging Face...")
        
        if self.dataset_name == "turk_corpus":
            raw_dataset = load_dataset("GEM/wiki_auto_asset_turk", split="test_turk")
            self.source_sentences = raw_dataset["source"]
            self.references = raw_dataset["references"] 
            
        elif self.dataset_name == "asset":
            raw_dataset = load_dataset("GEM/wiki_auto_asset_turk", split="test_asset")
            self.source_sentences = raw_dataset["source"]
            self.references = raw_dataset["references"]
            
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Choose 'turk_corpus' or 'asset'.")

        self.references = list(map(list, zip(*self.references)))
        print(f"✅ Loaded {len(self.source_sentences)} test sentences.")

    @staticmethod
    def _calculate_sari_pure_python(orig: str, sys: str, refs: list) -> float:
        """Pure Python implementation of the SARI metric (Xu et al., 2016)."""
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(zip(*[words[i:] for i in range(n)]))

        sari_score = 0
        for n in range(1, 5):
            o_ngrams = get_ngrams(orig, n)
            s_ngrams = get_ngrams(sys, n)
            
            r_ngrams_list = [get_ngrams(r, n) for r in refs]
            r_ngrams_all = set().union(*r_ngrams_list)

            # 1. ADD ADDITION METRIC
            r_minus_o = r_ngrams_all.difference(o_ngrams)
            s_minus_o = s_ngrams.difference(o_ngrams)
            
            precision_add = len(s_minus_o.intersection(r_minus_o)) / max(len(s_minus_o), 1)
            recall_add_list = [len(s_minus_o.intersection(r_ref.difference(o_ngrams))) / max(len(r_ref.difference(o_ngrams)), 1) for r_ref in r_ngrams_list]
            recall_add = np.mean(recall_add_list) if recall_add_list else 0
            f1_add = (2 * precision_add * recall_add) / (precision_add + recall_add) if (precision_add + recall_add) > 0 else 0

            # 2. KEEP METRIC
            s_and_o = s_ngrams.intersection(o_ngrams)
            precision_keep = len(s_and_o.intersection(r_ngrams_all)) / max(len(s_and_o), 1)
            recall_keep_list = [len(s_and_o.intersection(r_ref.intersection(o_ngrams))) / max(len(r_ref.intersection(o_ngrams)), 1) for r_ref in r_ngrams_list]
            recall_keep = np.mean(recall_keep_list) if recall_keep_list else 0
            f1_keep = (2 * precision_keep * recall_keep) / (precision_keep + recall_keep) if (precision_keep + recall_keep) > 0 else 0

            # 3. DELETE METRIC
            o_minus_s = o_ngrams.difference(s_ngrams)
            precision_del_list = [len(o_minus_s.intersection(o_ngrams.difference(r_ref))) / max(len(o_minus_s), 1) for r_ref in r_ngrams_list]
            precision_del = np.mean(precision_del_list) if precision_del_list else 0
            recall_del_list = [len(o_minus_s.intersection(o_ngrams.difference(r_ref))) / max(len(o_ngrams.difference(r_ref)), 1) for r_ref in r_ngrams_list]
            recall_del = np.mean(recall_del_list) if recall_del_list else 0
            f1_del = (2 * precision_del * recall_del) / (precision_del + recall_del) if (precision_del + recall_del) > 0 else 0

            sari_score += (f1_add + f1_keep + f1_del) / 3.0

        return (sari_score / 4.0) * 100

    def compute_metrics(self, system_outputs: list) -> dict:
        if not self.source_sentences:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        print("🧮 Calculating academic metrics...")
        
        # 🩹 CRITICAL FIX: Ensure no predictions are completely blank or empty strings, 
        # which breaks internal bert-score sentence encoder logic maps.
        safe_outputs = []
        for out in system_outputs:
            if out is None or not str(out).strip():
                safe_outputs.append(".") # Safe placeholder fallback
            else:
                safe_outputs.append(str(out).strip())

        # 1. BLEU Score
        bleu = sacrebleu.corpus_bleu(safe_outputs, self.references)
        bleu_score = bleu.score

        # Transpose references back to [samples, references] for iteration rows
        refs_per_sample = list(zip(*self.references))
        refs_as_strings_list = [list(refs) for refs in refs_per_sample]

        # 2. SARI Score
        sari_scores = []
        for orig, sys, refs in zip(self.source_sentences, safe_outputs, refs_per_sample):
            sari_scores.append(self._calculate_sari_pure_python(orig, sys, list(refs)))
        avg_sari = np.mean(sari_scores)

        # 3. METEOR Score (Handles lemmatization/synonyms natively)
        print("🧪 Extracting lemmatized match states via METEOR...")
        meteor_results = self.meteor_metric.compute(predictions=safe_outputs, references=refs_as_strings_list)
        meteor_score = meteor_results["meteor"] * 100

        # 4. BERTScore (Tracks contextual semantic similarity vectors)
        print("🧠 Running contextual vector alignment via BERTScore...")
        bert_results = self.bertscore_metric.compute(
            predictions=safe_outputs, 
            references=refs_as_strings_list, 
            lang="en",
            model_type="distilbert-base-uncased"
        )
        avg_bertscore_f1 = np.mean(bert_results["f1"]) * 100

        # 5. Compression Ratio
        orig_lens = [len(s.split()) for s in self.source_sentences]
        sys_lens = [len(s.split()) for s in safe_outputs]
        compression_ratio = np.mean(sys_lens) / np.mean(orig_lens)

        return {
            "SARI": round(avg_sari, 4),
            "BLEU": round(bleu_score, 4),
            "METEOR": round(meteor_score, 4),
            "BERTScore": round(avg_bertscore_f1, 4),
            "Compression_Ratio": round(compression_ratio, 4)
        }