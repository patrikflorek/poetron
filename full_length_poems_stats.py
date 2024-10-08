"""
This script analyzes Hviezdoslav's poems extracted from `books_txt.zip` archive
to count words and tokens, generate statistics, and idenftify "long poems".
"""

import os
import shutil
import urllib.request
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utils import calculate_average, calculate_median, calculate_mode, calculate_std_dev

DATA_DIR = "data"
FULL_LENGTH_DATA_DIR = os.path.join("data", "full_length")
BOOKS_ZIP = os.path.join(DATA_DIR, "books_txt.zip")
LONG_POEM_THRESHOLD = 1024

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Milos/slovak-gpt-j-405M")


def main():
    """Main function to execute the script."""
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Download and extract the books archive if not already present
    if not os.path.exists(BOOKS_ZIP):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/patrikflorek/hviezdoslav/master/books_txt.zip",
            BOOKS_ZIP,
        )

    shutil.rmtree(FULL_LENGTH_DATA_DIR, ignore_errors=True)
    with zipfile.ZipFile(BOOKS_ZIP, "r") as zip_ref:
        zip_ref.extractall(FULL_LENGTH_DATA_DIR)

    # Get counts and print statistics
    counts = get_counts()
    print_stats(counts)

    # Create a DataFrame with the counts sorted by book and poem
    counts_df = pd.DataFrame(
        [
            (book, poem, word_count, token_count)
            for (book, poem), (word_count, token_count) in counts.items()
        ],
        columns=["book", "poem", "word_count", "token_count"],
    ).sort_values(by=["book", "poem"], ignore_index=True)

    # Summary statistics
    print(counts_df.describe())

    # Save count histograms
    counts_df["word_count"].hist()
    plt.title("Poem word counts histogram")
    plt.savefig("full_length_word_count_hist.png")

    counts_df["token_count"].hist()
    plt.title("Poem token counts histogram")
    plt.savefig("full_length_token_count_hist.png")

    # Identify and save "long poems"
    long_poems = get_long_poems(counts)
    print("\nLong poems:")
    for book, poem, token_count in sorted(long_poems):
        print(f"  {book} - {poem}: {token_count} tokens")

    # Save "long poems" to CSV for further processing
    print("\nSaving long poems to long_poems.csv...")
    long_poems_df = pd.DataFrame(
        long_poems, columns=["book", "poem", "token_count"]
    ).sort_values(by=["book", "poem"], ignore_index=True)
    long_poems_df.to_csv("long_poems.csv", index=False)

    print("\n\nAll done!")


def get_counts():
    """
    Get word and token counts for all poems in the dataset.

    Returns:
        dict: Dictionary with keys (book, poem) and values (word_count, token_count).
    """
    counts = {}
    for root, _, poems in os.walk(FULL_LENGTH_DATA_DIR):
        book = os.path.basename(root)
        print(f"Checking {book}...")
        for poem in poems:
            word_count = count_words(os.path.join(root, poem))
            print(f"  {poem}: {word_count} words", end="")

            token_count = count_tokens(os.path.join(root, poem))
            print(f" and {token_count} tokens")
            counts[(book, poem)] = word_count, token_count
    return counts


def count_words(filename):
    """
    Count the number of words in a file.

    Args:
        filename (str): Path to the file.

    Returns:
        int: Number of words in the file.
    """
    with open(filename, "r") as f:
        return len(f.read().split())


def count_tokens(filename):
    """
    Count the number of tokens in a file using the tokenizer.

    Args:
        filename (str): Path to the file.

    Returns:
        int: Number of tokens in the file.
    """
    with open(filename, "r") as f:
        tokens = tokenizer(f.read())
        return len(tokens["input_ids"])


def print_stats(counts):
    """
    Print various statistics based on word and token counts.

    Args:
        counts (dict): Dictionary with keys (book, poem) and values (word_count, token_count).
    """
    total_word_counts = sum([c[0] for c in counts.values()])
    total_token_counts = sum([c[1] for c in counts.values()])
    print(f"\nTotal word counts: {total_word_counts}")
    print(f"Total token counts: {total_token_counts}")

    print("\nTop 5 poems by token count:")
    for (book, poem), (word_count, token_count) in sorted(
        counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {book} - {poem}: {word_count} words and {token_count} tokens")

    print("\nBottom 5 files by token count:")
    for (book, poem), (word_count, token_count) in sorted(
        counts.items(), key=lambda x: x[1]
    )[:5]:
        print(f"  {book} - {poem}: {word_count} words and {token_count} tokens")

    # Calculate and print average, median, mode, and standard deviation of word and token counts
    word_counts = [c[0] for c in counts.values()]
    token_counts = [c[1] for c in counts.values()]

    word_count_average = calculate_average(word_counts)
    token_count_average = calculate_average(token_counts)
    print(f"\nAverage word count: {word_count_average}")
    print(f"Average token count: {token_count_average}")

    word_count_median = calculate_median(word_counts)
    token_count_median = calculate_median(token_counts)
    print(f"\nMedian word count: {word_count_median}")
    print(f"Median token count: {token_count_median}")

    word_count_mode = calculate_mode(word_counts)
    token_count_mode = calculate_mode(token_counts)
    print(f"\nMode word count: {word_count_mode}")
    print(f"Mode token count: {token_count_mode}")

    word_count_std_dev = calculate_std_dev(word_counts)
    token_count_std_dev = calculate_std_dev(token_counts)
    print(f"\nStandard deviation of word count: {word_count_std_dev}")
    print(f"Standard deviation of token count: {token_count_std_dev}")


def get_long_poems(counts):
    """
    Identify long poems based on token count.

    Args:
        counts (dict): Dictioanary with keys (book, poem) and values (word_count, token_count).

    Returns:
        list: List of tuples (book, poem, token_count) for "long poems".
    """
    return [
        (book, poem, token_count)
        for (book, poem), (_, token_count) in counts.items()
        if token_count > LONG_POEM_THRESHOLD
    ]


if __name__ == "__main__":
    main()
