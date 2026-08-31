import torch

from llm.preprocessing.embeddings import TokenEmbedding
from llm.preprocessing.positional import PositionalEmbedding


def main():
    print("Token + Positional Embedding Demo")
    print("Type 'exit' to quit.\n")

    while True:
        context_input = input("Context length: ")

        if context_input.lower() == "exit":
            break

        try:
            context_length = int(context_input)
            embedding_dim = int(
                input("Embedding dimension: ")
            )

            token_input = input(
                "Enter token IDs separated by spaces: "
            )

            token_ids = [
                int(token)
                for token in token_input.split()
            ]

            if len(token_ids) > context_length:
                print(
                    "Number of tokens cannot exceed "
                    "context length.\n"
                )
                continue

            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
            )

            token_embedding = TokenEmbedding(
                vocab_size=50257,
                embedding_dim=embedding_dim,
            )

            positional_embedding = PositionalEmbedding(
                context_length=context_length,
                embedding_dim=embedding_dim,
            )

            token_vectors = token_embedding(input_ids)
            position_vectors = positional_embedding(
                input_ids
            )

            combined = (
                token_vectors + position_vectors
            )

            print("\nToken embedding shape:")
            print(token_vectors.shape)

            print("\nPositional embedding shape:")
            print(position_vectors.shape)

            print("\nCombined embedding shape:")
            print(combined.shape)

            print("-" * 50)

        except ValueError:
            print("Please enter valid numbers.\n")


if __name__ == "__main__":
    main()