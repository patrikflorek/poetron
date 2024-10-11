"""
This script analyzes chunks of Hviezdoslav's poems to count tokens and generate statistics.
It creates the final dataset for the Slovak GPT-J model fine-tuning to Hviezdoslav's poem writing style. 
"""

import os
import random
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utils import calculate_average, calculate_median, calculate_mode, calculate_std_dev


CHUNKED_DATA_DIR = "data/chunked"
GPT_J_CONTEXT_LENGTH = 2048

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Milos/slovak-gpt-j-405M")


def main():
    """Main function to execute the script."""
    # Ensure chunked data directory exists
    if not os.path.exists(CHUNKED_DATA_DIR):
        print(
            "The chunked data directory does not exist. "
            "Please run the `create_chunked.py` script first."
        )
        return

    # Get counts and print statistics
    print("Calculating token counts...")
    counts = get_counts()
    print_stats(counts)

    # Create a DataFrame with the counts sorted by book and poem chunk
    counts_df = pd.DataFrame(
        [
            (book, poem_chunk, token_count)
            for (book, poem_chunk), token_count in counts.items()
        ],
        columns=["book", "poem_chunk", "token_count"],
    ).sort_values(by=["book", "poem_chunk"], ignore_index=True)

    # Summary statistics
    print(counts_df.describe())

    # Save count histogram
    counts_df["token_count"].hist()
    plt.title("Poem chunk token count histogram")
    plt.savefig("chunked_token_count_hist.png")

    # Create create and dataset for fine-tuning
    print("\nCreating the dataset for fine-tuning...")
    save_dataset()

    print("\n\nAll done!")


def get_counts():
    """
    Get token counts for each poem chunk.

    Returns:
        dict: Dictionary with token counts for each poem chunk.
    """
    counts = {}
    for root, _, chunks in os.walk(CHUNKED_DATA_DIR):
        book = os.path.basename(root)
        for chunk in chunks:
            token_count = count_tokens(os.path.join(root, chunk))
            counts[(book, chunk)] = token_count
    return counts


def count_tokens(filename):
    """
    Count the number of tokens in a poem chunk.

    Args:
        filename (str): Path to the poem chunk file.

    Returns:
        int: Number of tokens in the poem chunk.
    """
    with open(filename, "r") as f:
        tokens = tokenizer(
            "[POH] " + f.read()
        )  # Prepends 5 tokens (59, 48, 8143, 61, 221) to indicate the text is a Hviezdoslav's poem
        return len(tokens["input_ids"])


def print_stats(counts):
    """
    Print various statistics based on token counts.

    Args:
        counts(dict): Dictionary with keys (book, poem_chunk) and token_count values.
    """
    total_token_count = sum(counts.values())
    print("\nTotal token count:", total_token_count)

    print("\nTop 5 poem chunks by token count:")
    for (book, poem_chunk), token_count in sorted(
        counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {book} - {poem_chunk}: {token_count} tokens")

    print("\nBottom 5 poem chunks by token count:")
    for (book, poem_chunk), token_count in sorted(counts.items(), key=lambda x: x[1])[
        :5
    ]:
        print(f"  {book} - {poem_chunk}: {token_count} tokens")

    # Calculate average token count
    token_count_mean = calculate_average(list(counts.values()))
    print("\nAverage token count:", token_count_mean)

    # Calculate median token count
    token_count_median = calculate_median(list(counts.values()))
    print("\nMedian token count:", token_count_median)

    # Calculate mode token count
    token_count_mode = calculate_mode(list(counts.values()))
    print("\nMode token count:", token_count_mode)

    # Calculate token count standard deviation
    token_count_std_dev = calculate_std_dev(list(counts.values()))
    print("\nToken count standard deviation:", token_count_std_dev)


def save_dataset():
    """
    Save the dataset for fine-tuning the Slovak GPT-J model to Hviezdoslav's poem writing style.
    """
    all_chunk_filenames = []
    for root, _, chunk_filenames in os.walk(CHUNKED_DATA_DIR):
        book = os.path.basename(root)
        for chunk_filename in chunk_filenames:
            all_chunk_filenames.append((book, chunk_filename))

    all_chunk_filenames.sort()

    random.seed(42)
    random.shuffle(all_chunk_filenames)
    train_size = int(0.8 * len(all_chunk_filenames))
    print(
        f"Train size: {train_size}, Test size: {len(all_chunk_filenames) - train_size}"
    )
    train_chunk_filenames = all_chunk_filenames[:train_size]
    test_chunk_filenames = all_chunk_filenames[train_size:]

    with zipfile.ZipFile("poh_dataset.zip", "w") as zipf:
        for split, chunk_filenames in [
            ("train", train_chunk_filenames),
            ("test", test_chunk_filenames),
        ]:
            for book, chunk_filename in chunk_filenames:
                with open(
                    os.path.join(CHUNKED_DATA_DIR, book, chunk_filename), "r"
                ) as f:
                    prepended_poem_chunk = "[POH] " + f.read()
                    zipf.writestr(
                        f"{split}/{book}/{chunk_filename}", prepended_poem_chunk
                    )


if __name__ == "__main__":
    main()
