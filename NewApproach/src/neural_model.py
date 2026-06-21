import torch
import string
from transformers import AutoTokenizer, AutoModelForMaskedLM

class CoreVocabularyLogitsProcessor:
    def __init__(self, tokenizer, allowed_vocab: set, exempt_vocab: set):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.mask = torch.full((self.vocab_size,), float("-inf"))
        
        # Preserve structural markers
        for s_id in set(tokenizer.all_special_ids):
            self.mask[s_id] = 0.0
            
        # Map allowed vocabulary
        for word in allowed_vocab:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                self.mask[token_ids[0]] = 0.0
                
        # Map protected grammar stopwords
        for word in exempt_vocab:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                self.mask[token_ids[0]] = 0.0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.mask = self.mask.to(logits.device)
        return logits + self.mask


class MLMNeuralSimplifier:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "distilbert-base-uncased"):
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        
        print(f"🤖 Initializing Constrained MLM Architecture: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        print("⚙️ Embedding negative constraint matrix into generation engine...")
        self.logits_processor = CoreVocabularyLogitsProcessor(
            tokenizer=self.tokenizer, 
            allowed_vocab=self.allowed_vocab, 
            exempt_vocab=self.exempt_vocab
        )

    def _clean_token(self, token: str) -> str:
        return token.strip(string.punctuation).lower()

    def batch_simplify(self, sentences: list) -> list:
        all_sentence_words = [sent.split() for sent in sentences]
        global_mask_jobs = []  
        flat_input_strings = []
        job_counter = 0
        
        for s_idx, words in enumerate(all_sentence_words):
            for w_idx, original_word in enumerate(words):
                clean_word = self._clean_token(original_word)
                if not clean_word or clean_word in self.allowed_vocab or clean_word in self.exempt_vocab:
                    continue
                
                temp_words = list(words)
                temp_words[w_idx] = self.mask_token
                masked_str = " ".join(temp_words)
                
                global_mask_jobs.append((job_counter, s_idx, w_idx, original_word))
                flat_input_strings.append(masked_str)
                job_counter += 1
        
        if global_mask_jobs:
            print(f"⚡ Executing constrained decoding on {len(flat_input_strings)} tasks simultaneously...")
            batch_size = 64
            
            for i in range(0, len(flat_input_strings), batch_size):
                batch_texts = flat_input_strings[i:i+batch_size]
                batch_jobs = global_mask_jobs[i:i+batch_size]
                
                inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_logits = outputs.logits
                
                for job_idx, (job, input_ids) in enumerate(zip(batch_jobs, inputs.input_ids)):
                    _, s_idx, w_idx, original_word = job
                    
                    mask_position = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_position) == 0:
                        continue
                    target_token_index = mask_position[0].item()
                    
                    raw_token_logits = batch_logits[job_idx, target_token_index]
                    
                    # Apply logit masking
                    constrained_logits = self.logits_processor(raw_token_logits)
                    
                    best_token_id = torch.argmax(constrained_logits).item()
                    valid_replacement = self.tokenizer.decode([best_token_id]).strip().lower()
                    
                    if not valid_replacement or best_token_id == self.tokenizer.unk_token_id:
                        valid_replacement = self._clean_token(original_word)
                    
                    if original_word[0] in string.punctuation and not valid_replacement.startswith(original_word[0]):
                        valid_replacement = original_word[0] + valid_replacement
                    if original_word[-1] in string.punctuation and not valid_replacement.endswith(original_word[-1]):
                        valid_replacement = valid_replacement + original_word[-1]
                        
                    all_sentence_words[s_idx][w_idx] = valid_replacement
                    
        return [" ".join(words) for words in all_sentence_words]