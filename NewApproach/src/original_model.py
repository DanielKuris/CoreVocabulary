import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingSubstitutionModel:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "distilbert-base-uncased"):
        """
        Initializes an ultra-lean Constrained Lexical Substitution Engine.
        Matches the interface architecture of the Seq2Seq configuration.
        """
        # Clean and standardize inputs exactly like your Seq2Seq setup
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        
        print(f"🤖 Model D: Initializing Custom Embedding Substitution Model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedder = AutoModel.from_pretrained(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder.to(self.device)
        self.embedder.eval()

        print("⚙️ Embedding high-speed token matrix constraints into generation engine...")
        # Compile and L2-normalize the allowed vocabulary vector space matrix immediately at load time
        self.vocab_words, self.vocab_matrix = self._compile_and_normalize_matrix()

    def _compile_and_normalize_matrix(self) -> tuple:
        """
        Extracts embedding vectors for the allowed vocabulary space directly from the active model,
        pre-normalizing them to allow instant dot-product cosine calculations.
        """
        words = list(self.allowed_vocab)
        vectors = []
        
        # Extract vectors for each allowed word using the tokenizer and embedder
        for word in words:
            inputs = self.tokenizer(word, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                outputs = self.embedder(**inputs)
                # Mean-pooled token hidden state
                vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            vectors.append(vec)
            
        matrix = np.vstack(vectors)
        
        # Pre-normalize the matrix to transform downstream cosine calculation into a simple dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / np.where(norms == 0, 1, norms)
        
        return words, normalized_matrix

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Generates a contextualized vector for a single isolated out-of-vocabulary token."""
        inputs = self.tokenizer(word, return_tensors="pt", add_special_tokens=False).to(self.device)
        if inputs["input_ids"].shape[1] == 0:
            return None
            
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        # Instantly normalize the incoming target vector
        vector_norm = np.linalg.norm(vector)
        return vector / vector_norm if vector_norm > 0 else vector

    def simplify_sentence(self, sentence: str) -> str:
        """Transforms a single incoming sentence string using strict vector mapping constraints."""
        tokens = sentence.strip().split()
        transformed_tokens = []

        for token in tokens:
            # Inline strip removes trailing punctuation markers safely
            clean_token = token.lower().strip(".,!?\"';:-_()")
            
            # Strict Dual-Perimeter Check
            if clean_token in self.allowed_vocab or clean_token in self.exempt_vocab or not clean_token:
                transformed_tokens.append(token)
                continue
                
            token_vector = self._get_word_vector(clean_token)
            
            if token_vector is None or token_vector.size == 0:
                transformed_tokens.append(token)
                continue
                
            # Pre-normalized dot product matrix multiplication replaces slow scikit-learn calls
            similarities = np.dot(self.vocab_matrix, token_vector)
            
            best_match_idx = np.argmax(similarities)
            substituted_word = self.vocab_words[best_match_idx]
            
            # Surface casing preservation layout
            if token[0].isupper():
                substituted_word = substituted_word.title() if token.istitle() else substituted_word.upper()
                
            transformed_tokens.append(substituted_word)

        return " ".join(transformed_tokens)

    def batch_simplify(self, sentences: list) -> list:
        """
        Processes a list of sentences sequentially through the substitution mapping pipeline, 
        matching your Seq2Seq class signature exactly.
        """
        simplified_sentences = []
        for sent in sentences:
            simplified_sentences.append(self.simplify_sentence(sent))
        return simplified_sentences