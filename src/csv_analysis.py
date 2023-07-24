import csv
from collections import Counter
import matplotlib.pyplot as plt
import math
import arabic_reshaper
from bidi.algorithm import get_display

def count_words_in_column(csv_file, column_index):
    word_lengths = []
    min_length = float('inf')
    min_length_rows = []
    min_length_content = []
    max_length = 0
    max_length_rows = []
    max_length_content = []
    total_word_length = 0
    row_count = 0
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row
        for index, row in enumerate(reader, start=1):
            if len(row) > column_index:
                sentence = row[column_index]
                words = sentence.split()
                filtered_words = [word for word in words if len(word) >= 1 and word not in ['ها', 'های']]
                word_count = len(filtered_words)
                if word_count > 0:
                    word_lengths.append(word_count)
                    total_word_length += word_count
                    row_count += 1
                    if word_count < min_length:
                        min_length = word_count
                        min_length_rows = [index+1]
                        min_length_content = [row[0]]
                    elif word_count == min_length:
                        min_length_rows.append(index+1)
                        min_length_content.append(row[0])
                    if word_count > max_length:
                        max_length = word_count
                        max_length_rows = [index+1]
                        max_length_content = [row[0]]
                    elif word_count == max_length:
                        max_length_rows.append(index+1)
                        max_length_content.append(row[0])
    return (
        word_lengths, min_length, min_length_rows, min_length_content,
        max_length, max_length_rows, max_length_content,
        total_word_length, row_count
    )

def calculate_average(word_lengths, total_word_length, row_count):
    if row_count > 0:
        return total_word_length / row_count
    return 0

def calculate_standard_deviation(word_lengths, average):
    squared_diff_sum = 0
    for length in word_lengths:
        squared_diff_sum += (length - average) ** 2
    variance = squared_diff_sum / len(word_lengths)
    return math.sqrt(variance)

def find_most_repeated_lengths(word_lengths, num_lengths=3):
    count_dict = {}
    for length in word_lengths:
        if length in count_dict:
            count_dict[length] += 1
        else:
            count_dict[length] = 1
    sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    most_repeated_lengths = [length for length, count in sorted_counts[:num_lengths]]
    num_repeats = [count for length, count in sorted_counts[:num_lengths]]
    return most_repeated_lengths, num_repeats

def plot_word_length_distribution(word_lengths, save_file=None):
    word_counts = {}
    for length in word_lengths:
        if length in word_counts:
            word_counts[length] += 1
        else:
            word_counts[length] = 1
    x_values = list(word_counts.keys())
    y_values = list(word_counts.values())
    plt.bar(x_values, y_values)
    plt.xlabel('Number of Words per caption')
    plt.ylabel('Number of Captions')
    plt.title('Caption Length Distribution')
    if save_file:
        plt.savefig(save_file, dpi=300)  # Save the plot to a PNG file
    plt.show()

def plot_most_repeated_words(words, counts, save_file=None):
    # Plot the frequencies
    words = [get_display(arabic_reshaper.reshape(label)) for label in words]
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top Most Repeated Words')
    plt.xticks(rotation=45)
    # Save the figure as a PNG file
    plt.savefig(save_file, bbox_inches='tight')
    # Show the plot
    plt.show()


def find_most_repeated_words(csv_file_path, column_index, num_words=3):
    word_counts = Counter()

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            sentence = row[column_index]
            words = sentence.split()
            filtered_words = [word for word in words if len(word) >= 1 and word not in ['ها', 'های']]
            word_counts.update(filtered_words)

    most_repeated_words = word_counts.most_common(num_words)
    return most_repeated_words

# Main

csv_file = '../dataset/captions_all.csv'
column_index = 2

(
    word_lengths, min_length, min_length_rows, min_length_content,
    max_length, max_length_rows, max_length_content,
    total_word_length, row_count
) = count_words_in_column(csv_file, column_index)
average_word_length = calculate_average(word_lengths, total_word_length, row_count)
standard_deviation = calculate_standard_deviation(word_lengths, average_word_length)
most_repeated_lengths, num_repeats = find_most_repeated_lengths(word_lengths, num_lengths=5) # Top num_length most repeated word length

most_repeated_words = find_most_repeated_words(csv_file, column_index, num_words=10) # Top num_words most repeated words

# Extract the words and their counts
words, counts = zip(*most_repeated_words)
# Print the most repeated words and their frequencies
for word, count in most_repeated_words:
    print(f'{word}: {count} occurrences')

# Plot
plot_most_repeated_words(words, counts, save_file="../paper_text/top_repeated.png")
plot_word_length_distribution(word_lengths, save_file="../paper_text/length_distribution.png")

# Show information
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
