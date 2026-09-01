
import torch
import tiktoken

from llm.gpt.model import GPTModel
from llm.gpt.config import GPT_CONFIG_124M
from llm.decoding.sampling import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


class GPTPipeline:
    """
    Reusable inference pipeline for the GPT language model.

    Converts user text into tokens, runs the GPT model,
    generates new tokens, and converts them back to text.
    """

    def __init__(
        self,
        model_path=None,
        config=None,
        device=None,
    ):
        # Select device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load checkpoint first if available
        checkpoint = None

        if model_path is not None:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )

        # Use checkpoint config when available.
        # Otherwise use explicitly provided config or default config.
        if checkpoint is not None and checkpoint.get("config") is not None:
            self.config = checkpoint["config"]
        else:
            self.config = config or GPT_CONFIG_124M

        # Create model using the exact configuration
        self.model = GPTModel(self.config)

        # Load trained weights
        if checkpoint is not None:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        self.model.to(self.device)
        self.model.eval()

        # GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        top_k=None,
        eos_id=None,
    ):
        """
        Generate text from a user-provided prompt.

        Args:
            prompt: User input text.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k sampling.
            eos_id: Optional end-of-sequence token ID.

        Returns:
            Generated text.
        """

        # Convert prompt → token IDs
        token_ids = text_to_token_ids(
            prompt,
            self.tokenizer,
        ).to(self.device)

        # Generate new tokens
        output_ids = generate(
            model=self.model,
            idx=token_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.config["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

        # Convert token IDs → text
        output_text = token_ids_to_text(
            output_ids,
            self.tokenizer,
        )

        return output_text

