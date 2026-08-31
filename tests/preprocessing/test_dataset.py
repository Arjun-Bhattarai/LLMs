from pathlib import Path

from llm.preprocessing.dataset import create_data_loader


DATA_PATH = Path("data/the-verdict.txt")


def main():
    text = DATA_PATH.read_text(encoding="utf-8")

    print("GPT Dataset / DataLoader")
    print("Type 'exit' to quit.\n")

    while True:
        batch_input = input("Batch size: ")

        if batch_input.lower() == "exit":
            break

        try:
            batch_size = int(batch_input)

            max_length = int(
                input("Context length: ")
            )

            stride = int(
                input("Stride: ")
            )

            if batch_size <= 0 or max_length <= 0 or stride <= 0:
                print("Values must be greater than 0.\n")
                continue

            data_loader = create_data_loader(
                text=text,
                batch_size=batch_size,
                max_length=max_length,
                stride=stride,
                shuffle=False,
            )

            print("\nNumber of samples:", len(data_loader.dataset))

            inputs, targets = next(iter(data_loader))

            print("\nInput IDs:")
            print(inputs)

            print("\nTarget IDs:")
            print(targets)

            print("\nInput shape:")
            print(inputs.shape)

            print("\nTarget shape:")
            print(targets.shape)

            print("\nNotice:")
            print("Target is shifted one token to the right.")

            print("-" * 50)

        except ValueError:
            print("Please enter valid integers.\n")


if __name__ == "__main__":
    main()