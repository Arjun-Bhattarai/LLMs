import re
from typing import Dict, List


class Tokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()

        return [
            self.str_to_int.get(
                token,
                self.str_to_int["<UNK>"]
            )
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> str:
        return " ".join(
            self.int_to_str[i]
            for i in ids
        )


def build_vocabulary(
    text: str,
    special_tokens: List[str] | None = None,
) -> Dict[str, int]:

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()

    vocab = sorted(set(tokens))

    if special_tokens:
        vocab.extend(
            token for token in special_tokens
            if token not in vocab
        )

    return {
        token: index
        for index, token in enumerate(vocab)
    }