
import torch
import tiktoken

from llm.decoding.sampling import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from llm.weights.gpt2_loader import load_pretrained_gpt2


class GPT2Pipeline:
    """
    Reusable inference interface for a pretrained GPT-2 model.

    Handles:
    - GPT-2 model loading
    - Tokenization
    - Text generation
    - Decoding
    """

    def __init__(
        self,
        model_size="124M",
        models_dir="gpt2",
        device=None,
    ):
        # Select device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained GPT-2 model
        self.model = load_pretrained_gpt2(
            model_size=model_size,
            models_dir=models_dir,
            device=self.device,
        )

        # GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Store context length
        self.context_length = (
            self.model.pos_emb.weight.shape[0]
        )

    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
        eos_id=None,
    ):
        """
        Generate text from a user prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Number of highest-probability tokens considered.
            eos_id: Optional end-of-sequence token ID.

        Returns:
            Generated text.
        """

        # Prompt → token IDs
        token_ids = text_to_token_ids(
            prompt,
            self.tokenizer,
        ).to(self.device)

        # Generate tokens
        with torch.no_grad():
            output_ids = generate(
                model=self.model,
                idx=token_ids,
                max_new_tokens=max_new_tokens,
                context_size=self.context_length,
                temperature=temperature,
                top_k=top_k,
                eos_id=eos_id,
            )

        # Token IDs → text
        return token_ids_to_text(
            output_ids,
            self.tokenizer,
        )

