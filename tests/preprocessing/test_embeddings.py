import torch

from llm.preprocessing.embeddings import TokenEmbedding


def main():
    print("Token Embedding Demo")
    print("Type 'exit' to quit.\n")

    while True:
        vocab_input = input("Enter vocabulary size: ")

        if vocab_input.lower() == "exit":
            break

        try:
            vocab_size = int(vocab_input)

            if vocab_size <= 0:
                print("Vocabulary size must be greater than 0.\n")
                continue

            embedding_input = input("Enter embedding dimension: ")

            if embedding_input.lower() == "exit":
                break

            embedding_dim = int(embedding_input)

            if embedding_dim <= 0:
                print("Embedding dimension must be greater than 0.\n")
                continue

            token_input = input(
                "Enter token IDs separated by spaces: "
            )

            if token_input.lower() == "exit":
                break

            token_ids = [int(token) for token in token_input.split()]

            # Make sure every token ID is valid
            if any(token < 0 or token >= vocab_size for token in token_ids):
                print(
                    f"Token IDs must be between 0 and {vocab_size - 1}.\n"
                )
                continue

            input_ids = torch.tensor(
                token_ids,
                dtype=torch.long,
            )

            embedding_layer = TokenEmbedding(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
            )

            print("\nEmbedding matrix:")
            print(embedding_layer.weight)

            output = embedding_layer(input_ids)

            print("\nInput token IDs:")
            print(input_ids)

            print("\nEmbedding output:")
            print(output)

            print("\nOutput shape:")
            print(output.shape)

            print("-" * 50)

        except ValueError:
            print(
                "Invalid input. Please enter numbers only.\n"
            )


if __name__ == "__main__":
    main()