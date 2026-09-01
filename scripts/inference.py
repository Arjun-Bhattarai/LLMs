
from llm.inference.gpt2_pipeline import GPT2Pipeline


def main():
    print("Loading GPT-2...")

    llm = GPT2Pipeline(
        model_size="124M",
        models_dir="gpt2",
    )

    print("GPT-2 loaded.")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() == "exit":
            break

        response = llm.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
        )

        print("\nGPT:", response)
        print()


if __name__ == "__main__":
    main()

