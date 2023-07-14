import csv
from collections import Counter
import matplotlib.pyplot as plt

def find_most_repeated_words(csv_file_path, column_index, num_words):
    word_counts = Counter()

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            sentence = row[column_index]
            words = sentence.split()
            word_counts.update(words)

    most_common_words = word_counts.most_common(num_words)
    return most_common_words

# Example usage
csv_file_path = 'Flicker30k-fa-en-all.csv' 
column_index = 2 
num_words = 10 

most_repeated_words = find_most_repeated_words(csv_file_path, column_index, num_words)

# Extract the words and their counts
words, counts = zip(*most_repeated_words)
# Print the most repeated words and their frequencies
for word, count in most_repeated_words:
    print(f'{word}: {count} occurrences')

# Plot the frequencies
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top Most Repeated Words')
plt.xticks(rotation=45)
# Save the figure as a PNG file
plt.savefig('word_frequencies.png', bbox_inches='tight')

# Show the plot
plt.show()
