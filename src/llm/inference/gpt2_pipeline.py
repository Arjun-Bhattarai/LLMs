
import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.gpt.model import GPTModel
from llm.decoding.sampling import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from llm.weights.gpt2_loader import load_pretrained_gpt2


class GPT2Pipeline:
    """
    Reusable inference interface for GPT-2.

    Supports:
    - Original pretrained GPT-2
    - Fine-tuned / continued-pretrained GPT-2 checkpoint
    - Tokenization
    - Text generation
    """

    def __init__(
        self,
        model_size="124M",
        models_dir="gpt2",
        checkpoint_path=None,
        device=None,
    ):
        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

                # Load model
        
        if checkpoint_path is not None:

            print(
                f"Loading trained GPT-2 checkpoint: "
                f"{checkpoint_path}"
            )

            # Your trained model uses the 124M architecture
            if model_size != "124M":
                raise ValueError(
                    "Checkpoint loading currently supports "
                    "only GPT-2 124M."
                )

            config = GPT_CONFIG_124M.copy()

            self.model = GPTModel(config)

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.model.to(self.device)
            self.model.eval()

        else:

            print("Loading original pretrained GPT-2...")

            self.model = load_pretrained_gpt2(
                model_size=model_size,
                models_dir=models_dir,
                device=self.device,
            )

                # Tokenizer
        
        self.tokenizer = tiktoken.get_encoding("gpt2")

                # Context length
        
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
        Generate text from a prompt.
        """

        # Prompt → token IDs
        token_ids = text_to_token_ids(
            prompt,
            self.tokenizer,
        ).to(self.device)

        # Generation
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

