import csv
import matplotlib.pyplot as plt
import math

def count_words_in_column(csv_file, column_index):
    word_lengths = []
    min_length = float('inf')
    num_min_length_rows = 0
    max_length = 0
    num_max_length_rows = 0
    total_word_length = 0
    row_count = 0
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row
        for index, row in enumerate(reader, start=1):
            if len(row) > column_index:
                sentence = row[column_index]
                words = sentence.split()
                word_count = len(words)
                if word_count > 0:
                    word_lengths.append(word_count)
                    total_word_length += word_count
                    row_count += 1
                    if word_count < min_length:
                        min_length = word_count
                        num_min_length_rows = 1
                    elif word_count == min_length:
                        num_min_length_rows += 1
                    if word_count > max_length:
                        max_length = word_count
                        num_max_length_rows = 1
                    elif word_count == max_length:
                        num_max_length_rows += 1
    return word_lengths, min_length, num_min_length_rows, max_length, num_max_length_rows, total_word_length, row_count

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

def count_words_draw(csv_file, column_index):
    word_lengths = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row
        for index, row in enumerate(reader, start=1):
            if index > 1 and len(row) > column_index:
                sentence = row[column_index]
                words = sentence.split()
                word_count = len(words)
                if word_count > 0:  # Exclude rows with zero word count
                    word_lengths.append(word_count)
    return word_lengths

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
    plt.xlabel('Number of Words')
    plt.ylabel('Number of Rows')
    plt.title('Word Length Distribution')
    if save_file:
        plt.savefig(save_file, dpi=300)  # Save the plot to a PNG file
    plt.show()


csv_file = 'Flicker30k-fa-en-all.csv'
column_index = 2
output_file = 'word_length_distribution.png'

word_lengths, min_length, num_min_length_rows, max_length, num_max_length_rows, total_word_length, row_count = count_words_in_column(csv_file, column_index)
average_word_length = calculate_average(word_lengths, total_word_length, row_count)
standard_deviation = calculate_standard_deviation(word_lengths, average_word_length)
most_repeated_lengths, num_repeats = find_most_repeated_lengths(word_lengths, num_lengths=5)

print(f"Minimum word length: {min_length}")
print(f"Number of rows with minimum word length: {num_min_length_rows}")
print(f"Maximum word length: {max_length}")
print(f"Number of rows with maximum word length: {num_max_length_rows}")
print(f"Average word length: {average_word_length:.2f}")
print(f"Standard deviation of word lengths: {standard_deviation:.2f}")
print("Most Repeated Word Lengths:")
for i, length in enumerate(most_repeated_lengths, start=1):
    print(f"{i}. Length: {length}, Repeats: {num_repeats[i-1]}")

word_lengths = count_words_draw(csv_file, column_index)
plot_word_length_distribution(word_lengths, save_file=output_file)