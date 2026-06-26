import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5SubwordTrie:
    def __init__(self, tokenizer, vocabulary: set):
        self.root = {}
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        variants = []
        for word in vocabulary:
            variants.extend([word, f" {word}"])
            
        if variants:
            encoded_variants = tokenizer(variants, add_special_tokens=False)["input_ids"]
            for token_ids in encoded_variants:
                self.insert(token_ids)

    def insert(self, token_ids: list):
        current = self.root
        for t_id in token_ids:
            if t_id not in current:
                current[t_id] = {}
            current = current[t_id]
        current['<end>'] = True

class DynamicT5Simplifier:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "t5-small", tokenizer=None, model=None):
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        self.full_universe = self.allowed_vocab.union(self.exempt_vocab)
        
        print(f"🤖 Initializing Constrained Subword T5 Engine ({model_name})...")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.model = model if model is not None else AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.trie = T5SubwordTrie(self.tokenizer, self.full_universe)
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _get_prefix_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor) -> list:
        current_node = self.trie.root
        allowed = [self.eos_id, self.pad_id]
        
        generated_tokens = input_ids.tolist()
        sep_idx = -1
        for idx in range(len(generated_tokens) - 1, -1, -1):
            if generated_tokens[idx] in [self.pad_id, self.eos_id, 0]:
                sep_idx = idx
                break
        active_chunk = generated_tokens[sep_idx + 1:]
            
        for t_id in active_chunk:
            if t_id in current_node:
                current_node = current_node[t_id]
            else:
                return list(set(list(self.trie.root.keys()) + allowed))
                
        valid_next_tokens = [k for k in current_node.keys() if k != '<end>']
        allowed.extend(valid_next_tokens)
        
        if '<end>' in current_node or current_node == self.trie.root:
            allowed.extend(list(self.trie.root.keys()))
            
        return list(set(allowed))

    def batch_simplify(self, sentences: list) -> list:
        prefix = "simplify: "
        prefixed_sentences = [prefix + sent for sent in sentences]
        batch_size = 64
        simplified_sentences = []
        
        for i in range(0, len(prefixed_sentences), batch_size):
            batch_texts = prefixed_sentences[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=64,
                    num_beams=5,
                    prefix_allowed_tokens_fn=self._get_prefix_allowed_tokens,
                    early_stopping=True
                )
                
            batch_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            simplified_sentences.extend(batch_decoded)
            
        return [sent.strip() for sent in simplified_sentences]