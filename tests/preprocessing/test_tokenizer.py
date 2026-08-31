from pathlib import Path

from llm.preprocessing.tokenizer import (
    Tokenizer,
    build_vocabulary,
)


DATA_PATH = Path("data/the-verdict.txt")


def main():
    raw_data = DATA_PATH.read_text(encoding="utf-8")

    vocab = build_vocabulary(
        raw_data,
        special_tokens=[
            "<|endoftext|>",
            "<UNK>",
        ],
    )

    tokenizer = Tokenizer(vocab)

    print("Simple Word-Level Tokenizer")
    print(f"Total characters: {len(raw_data)}")
    print(f"Vocabulary size: {len(vocab)}")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            break

        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print("\nEncoded:", ids)
        print("Number of tokens:", len(ids))
        print("Decoded:", decoded)
        print("-" * 50)


if __name__ == "__main__":
    main()