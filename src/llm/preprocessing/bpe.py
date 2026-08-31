import tiktoken


class GPT2Tokenizer:
    """GPT-2 BPE tokenizer using tiktoken."""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(
        self,
        text: str,
        allowed_special: set[str] | None = None,
    ) -> list[int]:
        if allowed_special is None:
            allowed_special = set()

        return self.tokenizer.encode(
            text,
            allowed_special=allowed_special,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab