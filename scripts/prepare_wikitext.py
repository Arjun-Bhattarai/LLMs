from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/wikitext-103")


def parquet_to_text(pattern, output_file):
    files = sorted(DATA_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for: {pattern}")

    texts = []

    for file in files:
        print(f"Reading {file}")

        df = pd.read_parquet(file)

        # WikiText uses a "text" column
        texts.extend(
            text
            for text in df["text"].astype(str)
            if text.strip()
        )

    text = "\n\n".join(texts)

    output_path = OUTPUT_DIR / output_file

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Created: {output_path}")
    print(f"Characters: {len(text):,}")


if __name__ == "__main__":
    parquet_to_text(
        "train-*.parquet",
        "train.txt",
    )

    parquet_to_text(
        "validation-*.parquet",
        "validation.txt",
    )