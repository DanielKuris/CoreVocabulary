import numpy as np
from datasets import load_dataset
import datasets
import sacrebleu
import evaluate
from collections import Counter
import torch
import sys
import logging
import nltk
import pandas as pd

# 🆕 Dynamic structural link to your centralized matrix config
import configuration as config

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
datasets.logging.set_verbosity_error()

nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4", quiet=True)

class SimplificationEvaluator:
    print("📥 Initializing METEOR and BERTScore shared evaluation layers...")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")

    def __init__(self, dataset_name: str = "turk_corpus"):
        self.dataset_name = dataset_name.lower().strip()
        self.source_sentences = []
        self.references = []  

    def load_data(self):
        print(f"📦 Loading '{self.dataset_name}'...")
        
        if self.dataset_name == "turk_corpus":
            raw_dataset = load_dataset("GEM/wiki_auto_asset_turk", split="test_turk")
            self.source_sentences = raw_dataset["source"]
            self.references = raw_dataset["references"] 
            self.references = list(map(list, zip(*self.references)))
            
        elif self.dataset_name == "asset":
            raw_dataset = load_dataset("GEM/wiki_auto_asset_turk", split="test_asset")
            self.source_sentences = raw_dataset["source"]
            self.references = raw_dataset["references"]
            self.references = list(map(list, zip(*self.references)))
            
        elif self.dataset_name == "sick":
            url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_test_annotated.txt"
            try:
                df = pd.read_csv(url, sep="\t")
            except Exception as e:
                backup_url = "https://raw.githubusercontent.com/asobg/RTE-Dataset-parsing/master/SICK/SICK_test_annotated.txt"
                df = pd.read_csv(backup_url, sep="\t")
                
            # Convert raw text matrix tracking arrays
            raw_sources = df["sentence_A"].astype(str).tolist()
            raw_targets = df["sentence_B"].astype(str).tolist()
            
            # 🆕 DYNAMIC SLICING CHECKER: Resolves slice bounds or defaults to absolute length
            slice_limit = getattr(config, "SICK_SLICING", None)
            
            if isinstance(slice_limit, int):
                self.source_sentences = raw_sources[:slice_limit]
                self.references = [raw_targets[:slice_limit]]
            else:
                self.source_sentences = raw_sources
                self.references = [raw_targets]
            
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Choose 'turk_corpus', 'asset', or 'sick'.")

        print(f"✅ Loaded {len(self.source_sentences)} test sentences.")

    @staticmethod
    def _calculate_sari_pure_python(orig: str, sys: str, refs: list) -> float:
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(zip(*[words[i:] for i in range(n)]))

        sari_score = 0
        for n in range(1, 5):
            o_ngrams = get_ngrams(orig, n)
            s_ngrams = get_ngrams(sys, n)
            
            r_ngrams_list = [get_ngrams(r, n) for r in refs]
            r_ngrams_all = set().union(*r_ngrams_list)

            r_minus_o = r_ngrams_all.difference(o_ngrams)
            s_minus_o = s_ngrams.difference(o_ngrams)
            
            with np.errstate(invalid='ignore'):
                precision_add = len(s_minus_o.intersection(r_minus_o)) / max(len(s_minus_o), 1)
                recall_add_list = [len(s_minus_o.intersection(r_ref.difference(o_ngrams))) / max(len(r_ref.difference(o_ngrams)), 1) for r_ref in r_ngrams_list]
                recall_add = np.mean(recall_add_list) if recall_add_list else 0
                f1_add = (2 * precision_add * recall_add) / (precision_add + recall_add) if (precision_add + recall_add) > 0 else 0

                s_and_o = s_ngrams.intersection(o_ngrams)
                precision_keep = len(s_and_o.intersection(r_ngrams_all)) / max(len(s_and_o), 1)
                recall_keep_list = [len(s_and_o.intersection(r_ref.intersection(o_ngrams))) / max(len(r_ref.intersection(o_ngrams)), 1) for r_ref in r_ngrams_list]
                recall_keep = np.mean(recall_keep_list) if recall_keep_list else 0
                f1_keep = (2 * precision_keep * recall_keep) / (precision_keep + recall_keep) if (precision_keep + recall_keep) > 0 else 0

                o_minus_s = o_ngrams.difference(s_ngrams)
                precision_del_list = [len(o_minus_s.intersection(o_ngrams.difference(r_ref))) / max(len(o_minus_s), 1) for r_ref in r_ngrams_list]
                precision_del = np.mean(precision_del_list) if precision_del_list else 0
                recall_del_list = [len(o_minus_s.intersection(o_ngrams.difference(r_ref))) / max(len(o_ngrams.difference(r_ref)), 1) for r_ref in r_ngrams_list]
                recall_del = np.mean(recall_del_list) if recall_del_list else 0
                f1_del = (2 * precision_del * recall_del) / (precision_del + recall_del) if (precision_del + recall_del) > 0 else 0

            sari_score += (f1_add + f1_keep + f1_del) / 3.0

        return (sari_score / 4.0) * 100

    @staticmethod
    def _calculate_jaccard(sent1: str, sent2: str) -> float:
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / max(len(union), 1)

    def compute_metrics(self, system_outputs: list) -> dict:
        if not self.source_sentences:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        print("🧮 Calculating academic metrics...")
        
        safe_outputs = []
        for out in system_outputs:
            if out is None or not str(out).strip():
                safe_outputs.append(".") 
            else:
                safe_outputs.append(str(out).strip())

        bleu = sacrebleu.corpus_bleu(safe_outputs, self.references)
        bleu_score = bleu.score

        refs_per_sample = list(zip(*self.references))
        refs_as_strings_list = [list(refs) for refs in refs_per_sample]

        sari_scores = []
        jaccard_scores = []
        
        for orig, sys, refs in zip(self.source_sentences, safe_outputs, refs_per_sample):
            sari_scores.append(self._calculate_sari_pure_python(orig, sys, list(refs)))
            jaccard_scores.append(self._calculate_jaccard(orig, sys))
            
        avg_sari = np.mean(sari_scores)
        avg_jaccard = np.mean(jaccard_scores) * 100

        print("🧪 Extracting lemmatized match states via METEOR...")
        meteor_results = self.__class__.meteor_metric.compute(predictions=safe_outputs, references=refs_as_strings_list)
        meteor_score = meteor_results["meteor"] * 100

        print("🧠 Running contextual vector alignment via BERTScore...")
        bert_results = self.__class__.bertscore_metric.compute(
            predictions=safe_outputs, 
            references=self.source_sentences, 
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False
        )
        avg_cosine_similarity = np.mean(bert_results["precision"]) * 100
        
        bert_ref_results = self.__class__.bertscore_metric.compute(
            predictions=safe_outputs, 
            references=refs_as_strings_list, 
            lang="en",
            model_type="distilbert-base-uncased"
        )
        avg_bertscore_f1 = np.mean(bert_ref_results["f1"]) * 100

        orig_lens = [len(s.split()) for s in self.source_sentences]
        sys_lens = [len(s.split()) for s in safe_outputs]
        compression_ratio = np.mean(sys_lens) / np.mean(orig_lens)

        return {
            "SARI": round(avg_sari, 4),
            "BLEU": round(bleu_score, 4),
            "METEOR": round(meteor_score, 4),
            "BERTScore": round(avg_bertscore_f1, 4),
            "Jaccard_Similarity": round(avg_jaccard, 4),
            "Cosine_Similarity": round(avg_cosine_similarity, 4),
            "Compression_Ratio": round(compression_ratio, 4)
        }