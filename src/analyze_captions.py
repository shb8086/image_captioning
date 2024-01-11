"""
File: analyze_captions.py
@author: Shima Baniadamdizaj
@email: baniadam.shima@gmail.com

This script analyzes captions from a CSV file, providing insights into word lengths, distribution, and most repeated words.

Dependencies:
- csv
- collections.Counter
- matplotlib.pyplot
- bidi.algorithm
"""

import csv
from collections import Counter
import matplotlib.pyplot as plt
from bidi.algorithm import get_display

def load_captions(csv_file, column_index):
    """Load captions from a CSV file."""
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row
        return [row[column_index] for row in reader if len(row) > column_index]

def clean_and_tokenize(caption):
    """Clean and tokenize a caption."""
    return [word for word in caption.split() if len(word) >= 1 and word not in ['ها', 'های']]

def analyze_captions(csv_file, column_index):
    """Analyze captions from a CSV file."""
    captions = load_captions(csv_file, column_index)

    word_lengths = [len(clean_and_tokenize(caption)) for caption in captions]
    total_word_length = sum(word_lengths)
    row_count = len(captions)

    if row_count == 0:
        return [], 0, [], [], 0, [], [], 0, 0

    min_length = min(word_lengths)
    min_length_rows = [i+1 for i, length in enumerate(word_lengths) if length == min_length]
    min_length_content = [captions[i] for i in min_length_rows]

    max_length = max(word_lengths)
    max_length_rows = [i+1 for i, length in enumerate(word_lengths) if length == max_length]
    max_length_content = [captions[i] for i in max_length_rows]

    return (
        word_lengths, min_length, min_length_rows, min_length_content,
        max_length, max_length_rows, max_length_content,
        total_word_length, row_count
    )

def calculate_average(word_lengths, total_word_length, row_count):
    """Calculate the average word length."""
    return total_word_length / row_count if row_count > 0 else 0

def calculate_standard_deviation(word_lengths, average):
    """Calculate the standard deviation of word lengths."""
    variance = sum((length - average) ** 2 for length in word_lengths) / len(word_lengths)
    return variance ** 0.5

def find_most_repeated_lengths(word_lengths, num_lengths=3):
    """Find the most repeated word lengths."""
    count_dict = Counter(word_lengths)
    most_repeated_lengths = [length for length, _ in count_dict.most_common(num_lengths)]
    num_repeats = [count for _, count in count_dict.most_common(num_lengths)]
    return most_repeated_lengths, num_repeats

def find_most_repeated_words(captions, num_words=3):
    """Find the most repeated words."""
    word_counts = Counter(word for caption in captions for word in clean_and_tokenize(caption))
    return word_counts.most_common(num_words)

def plot_most_repeated_words(words, counts, save_file=None):
    """Plot the frequencies of the most repeated words."""
    words = [get_display(word) for word in words]
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top Most Repeated Words')
    plt.xticks(rotation=45)
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()

def plot_word_length_distribution(word_lengths, save_file=None):
    """Plot the distribution of word lengths."""
    plt.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 1), align='left')
    plt.xlabel('Number of Words per Caption')
    plt.ylabel('Number of Captions')
    plt.title('Caption Length Distribution')
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()

if __name__ == "__main__":
    csv_file = '../dataset/captions_all.csv'
    column_index = 2

    (
        word_lengths, min_length, min_length_rows, min_length_content,
        max_length, max_length_rows, max_length_content,
        total_word_length, row_count
    ) = analyze_captions(csv_file, column_index)

    average_word_length = calculate_average(word_lengths, total_word_length, row_count)
    standard_deviation = calculate_standard_deviation(word_lengths, average_word_length)

    most_repeated_lengths, num_repeats = find_most_repeated_lengths(word_lengths, num_lengths=5)
    most_repeated_words = find_most_repeated_words(load_captions(csv_file, column_index), num_words=10)

    words, counts = zip(*most_repeated_words)
    for word, count in most_repeated_words:
        print(f'{word}: {count} occurrences')

    plot_most_repeated_words(words, counts, save_file="../paper_text/top_repeated.png")
    plot_word_length_distribution(word_lengths, save_file="../paper_text/length_distribution.png")

    print(f"\nMinimum word length: {min_length}")
    print(f"\nNumber of rows with minimum word length: {len(min_length_rows)}")
    print(f"\nRow numbers with minimum word length: {min_length_rows}")
    print(f"\nContent of the first column for minimum word length: {min_length_content}")
    print(f"\nMaximum word length: {max_length}")
    print(f"\nNumber of rows with maximum word length: {len(max_length_rows)}")
    print(f"\nRow numbers with maximum word length: {max_length_rows}")
    print(f"\nContent of the first column for maximum word length: {max_length_content}")
    print(f"\nAverage word length: {average_word_length:.2f}")
    print(f"\nStandard deviation of word lengths: {standard_deviation:.2f}")
    print("\nMost Repeated Word Lengths:")
    for i, length in enumerate(most_repeated_lengths, start=1):
        print(f"{i}. Length: {length}, Repeats: {num_repeats[i-1]}")
