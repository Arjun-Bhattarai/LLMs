from llm.preprocessing.bpe import GPT2Tokenizer


def main():
    tokenizer = GPT2Tokenizer()

    print("GPT-2 BPE Tokenizer")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            break

        token_ids = tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

        decoded = tokenizer.decode(token_ids)

        print("\nToken IDs:")
        print(token_ids)

        print("\nNumber of tokens:")
        print(len(token_ids))

        print("\nDecoded text:")
        print(decoded)

        print("-" * 50)


if __name__ == "__main__":
    main()