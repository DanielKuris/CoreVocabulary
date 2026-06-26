import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class VocabularyTrie:
    def __init__(self, vocabulary: set):
        self.root = TrieNode()
        for word in vocabulary:
            self.insert(word)
            
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True  # Fixed leaf node tracking assignment


class CharacterTrieNeuralSimplifier:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "google/byt5-small", tokenizer=None, model=None):
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        self.full_universe = self.allowed_vocab.union(self.exempt_vocab)
        
        print(f"🤖 Initializing Positive Constrained Character-Level Architecture ({model_name})...")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.model = model if model is not None else AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.trie = VocabularyTrie(self.full_universe)
        self.space_char = " "
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Precompute space token ID and cache character encodings
        self.space_ids = self.tokenizer.encode(self.space_char, add_special_tokens=False)
        self.char_to_token_ids = {}

    def _get_prefix_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor) -> list:
        decoded_history = self.tokenizer.decode(input_ids, skip_special_tokens=True).lower()
        words = decoded_history.split(self.space_char)
        current_word_fragment = words[-1] if words else ""
        
        allowed_next_tokens = [self.eos_token_id, self.pad_token_id]
        
        node = self.trie.root
        match_possible = True
        for char in current_word_fragment:
            if char in node.children:
                node = node.children[char]
            else:
                match_possible = False
                break
                
        if match_possible:
            for next_char in node.children.keys():
                if next_char not in self.char_to_token_ids:
                    self.char_to_token_ids[next_char] = self.tokenizer.encode(next_char, add_special_tokens=False)
                allowed_next_tokens.extend(self.char_to_token_ids[next_char])
                
            if node.is_end_of_word or current_word_fragment in self.full_universe:
                allowed_next_tokens.extend(self.space_ids)
        else:
            allowed_next_tokens.extend(self.space_ids)
            
        return list(set(allowed_next_tokens))

    def batch_simplify(self, sentences: list) -> list:
        prefix = "simplify: "
        prefixed_sentences = [prefix + sent for sent in sentences]
        batch_size = 8
        simplified_sentences = []
        
        for i in range(0, len(prefixed_sentences), batch_size):
            batch_texts = prefixed_sentences[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=128,
                    num_beams=5,
                    prefix_allowed_tokens_fn=self._get_prefix_allowed_tokens,
                    early_stopping=True
                )
                
            batch_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            simplified_sentences.extend(batch_decoded)
            
        return [sent.strip() for sent in simplified_sentences]