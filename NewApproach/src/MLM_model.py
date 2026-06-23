import torch
import string
from transformers import AutoTokenizer, AutoModelForMaskedLM

class CoreVocabularyLogitsProcessor:
    def __init__(self, tokenizer, allowed_vocab: set, exempt_vocab: set):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.mask = torch.full((self.vocab_size,), float("-inf"))
        
        # Preserve structural special tokens
        for s_id in set(tokenizer.all_special_ids):
            if s_id < self.vocab_size:
                self.mask[s_id] = 0.0
            
        self.single_token_words = {}
        full_vocab = allowed_vocab.union(exempt_vocab)
        
        for word in full_vocab:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                self.mask[token_ids[0]] = 0.0
                self.single_token_words[token_ids[0]] = word

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.mask = self.mask.to(logits.device)
        return logits + self.mask


class MLMNeuralSimplifier:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "distilbert-base-uncased"):
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        self.full_universe = self.allowed_vocab.union(self.exempt_vocab)
        
        print(f"🤖 Initializing Constrained MLM Architecture: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        self.logits_processor = CoreVocabularyLogitsProcessor(
            tokenizer=self.tokenizer, 
            allowed_vocab=self.allowed_vocab, 
            exempt_vocab=self.exempt_vocab
        )
        
        # Pre-compile multi-token target representations to avoid runtime overhead
        self.multi_token_words = {}
        for word in self.full_universe:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(ids) > 1:
                self.multi_token_words[word] = ids

    def _clean_token(self, token: str) -> str:
        return token.strip(string.punctuation).lower()

    def batch_simplify(self, sentences: list) -> list:
        cleaned_sentences = []
        
        # Absolute structural baseline word for fallback guarantees
        absolute_fallback_word = list(self.allowed_vocab)[0] if self.allowed_vocab else "the"
        
        for sent in sentences:
            words = sent.split()
            for w_idx, original_word in enumerate(words):
                clean_word = self._clean_token(original_word)
                
                # Check vocabulary compliance
                if not clean_word or clean_word in self.full_universe:
                    continue
                
                # Stage 1: Evaluate single-token alternatives via logits mask
                temp_words = list(words)
                temp_words[w_idx] = self.mask_token
                masked_str = " ".join(temp_words)
                
                inputs = self.tokenizer(masked_str, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0]
                    
                mask_pos = (inputs.input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                target_idx = mask_pos[0].item()
                
                single_logits = self.logits_processor(logits[target_idx])
                best_single_token_id = torch.argmax(single_logits).item()
                best_single_score = single_logits[best_single_token_id].item()
                
                best_word = self.logits_processor.single_token_words.get(best_single_token_id, None)
                best_score = best_single_score if (best_word and best_single_token_id != self.tokenizer.unk_token_id) else float("-inf")
                
                # Stage 2: Evaluate multi-token candidates using length-normalized joint probability
                for multi_word, token_ids in self.multi_token_words.items():
                    multi_masks = [self.mask_token] * len(token_ids)
                    multi_temp_words = list(words)
                    multi_temp_words[w_idx] = " ".join(multi_masks)
                    multi_masked_str = " ".join(multi_temp_words)
                    
                    m_inputs = self.tokenizer(multi_masked_str, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        m_outputs = self.model(**m_inputs)
                        m_logits = m_outputs.logits[0]
                        
                    m_mask_positions = (m_inputs.input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
                    if len(m_mask_positions) != len(token_ids):
                        continue
                        
                    word_score = 0.0
                    for pos_idx, target_token_id in zip(m_mask_positions, token_ids):
                        log_probs = torch.log_softmax(m_logits[pos_idx.item()], dim=-1)
                        word_score += log_probs[target_token_id].item()
                        
                    normalized_score = word_score / len(token_ids)
                    
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_word = multi_word
                
                # Strict vocabulary boundary guarantee fallback
                if not best_word or best_word.strip() == "":
                    best_word = absolute_fallback_word
                
                # Maintain original formatting parameters
                if original_word[0] in string.punctuation and not best_word.startswith(original_word[0]):
                    best_word = original_word[0] + best_word
                if original_word[-1] in string.punctuation and not best_word.endswith(original_word[-1]):
                    best_word = best_word + original_word[-1]
                    
                words[w_idx] = best_word
                
            cleaned_sentences.append(" ".join(words))
            
        return cleaned_sentences