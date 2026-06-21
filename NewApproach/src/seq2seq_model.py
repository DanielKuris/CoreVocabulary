import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessor, LogitsProcessorList

class FastSeq2SeqVocabularyLogitsProcessor(LogitsProcessor):
    """
    Optimized Vectorized Constrained Decoding Processor for T5.
    Compiles allowed sub-word combinations at initialization to run instant 
    Tensor operations during runtime, avoiding heavy string-matching loops.
    """
    def __init__(self, tokenizer, model, allowed_vocab: set, exempt_vocab: set):
        self.tokenizer = tokenizer
        
        # Capture the true dynamic output head size directly from the model config
        self.vocab_size = model.config.vocab_size # This resolves exactly to 32100
        
        # Initialize the mask matrix to match the model's exact score tensor width
        self.static_mask = torch.full((self.vocab_size,), float("-inf"))
        
        # Unban special infrastructure tokens (EOS, PAD, etc.)
        for s_id in set(tokenizer.all_special_ids):
            if s_id < self.vocab_size:
                self.static_mask[s_id] = 0.0
            
        # Combine words into a unified target set
        full_allowed_words = allowed_vocab.union(exempt_vocab)
        
        # Extract every individual sub-token piece used by T5 to represent allowed words
        allowed_token_ids = set()
        for word in full_allowed_words:
            for variant in [word, f" {word}"]:
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                for t_id in token_ids:
                    if t_id < self.vocab_size:
                        allowed_token_ids.add(t_id)
                        
        # Unban the entire vocabulary sub-space simultaneously via vectorized assignment
        for t_id in allowed_token_ids:
            self.static_mask[t_id] = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Executes instant batch-wide logit surgery using parallel matrix addition.
        """
        # Move the mask to the active tensor device (GPU/CPU) dynamically
        mask = self.static_mask.to(scores.device)
        
        # scores shape: [batch_size * num_beams, active_vocab_size (32100)]
        # Direct parallel broadcasting element-wise addition across dimension 1
        return scores + mask


class Seq2SeqNeuralSimplifier:
    def __init__(self, vocabulary: set, exempt_vocabulary: set = None, model_name: str = "t5-small"):
        self.allowed_vocab = {word.lower().strip() for word in vocabulary}
        self.exempt_vocab = {word.lower().strip() for word in exempt_vocabulary} if exempt_vocabulary else set()
        
        print(f"🤖 Model B: Initializing Custom Constrained Seq2Seq Architecture ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print("⚙️ Embedding high-speed token matrix constraints into generation engine...")
        # Note: We now pass self.model directly into the processor to sample its config dimensions
        self.logits_processor_hook = FastSeq2SeqVocabularyLogitsProcessor(
            tokenizer=self.tokenizer,
            model=self.model,
            allowed_vocab=self.allowed_vocab,
            exempt_vocab=self.exempt_vocab
        )

    def batch_simplify(self, sentences: list) -> list:
        prefix = "simplify: "
        prefixed_sentences = [prefix + sent for sent in sentences]
        
        # Use small batches to maximize GPU memory efficiency and processing speeds
        batch_size = 16 
        simplified_sentences = []
        processors = LogitsProcessorList([self.logits_processor_hook])
        
        for i in range(0, len(prefixed_sentences), batch_size):
            batch_texts = prefixed_sentences[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=64,
                    num_beams=5,
                    logits_processor=processors,
                    early_stopping=True
                )
                
            batch_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            simplified_sentences.extend(batch_decoded)
            
        return [sent.strip() for sent in simplified_sentences]