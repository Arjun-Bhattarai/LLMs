
from llm.inference.gpt2_pipeline import GPT2Pipeline

# Configuration

# True  → use your WikiText-trained checkpoint
# False → use original OpenAI GPT-2
USE_TRAINED_MODEL = False

CHECKPOINT_PATH = (
    "checkpoints/gpt2_wikitext.pt"
)


def main():

    print("=" * 50)

    if USE_TRAINED_MODEL:

        print("Loading WikiText-trained GPT-2...")

        llm = GPT2Pipeline(
            model_size="124M",
            models_dir="gpt2",
            checkpoint_path=CHECKPOINT_PATH,
        )

        print("WikiText-trained GPT-2 loaded.")

    else:

        print("Loading original GPT-2...")

        llm = GPT2Pipeline(
            model_size="124M",
            models_dir="gpt2",
        )

        print("Original GPT-2 loaded.")

    print("=" * 50)
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

        print("\nYour LLM:", response)
        print()


if __name__ == "__main__":
    main()

