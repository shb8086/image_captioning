import csv
import re

def contains_alpha_numeric(text):
    return bool(re.search(r'[a-zA-Z0-9]', text))

def find_rows_with_alpha_numeric(csv_file, column_index):
    rows_with_alpha_numeric = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if column_index < len(row):
                cell = row[column_index]
                words = cell.split()  # Split sentence into words
                for word in words:
                    if contains_alpha_numeric(word):
                        rows_with_alpha_numeric.append(row_number)
                        break  # Only store the row number once per row

    return rows_with_alpha_numeric

csv_file = '../dataset/captions_all.csv'  
column_index = 2

rows = find_rows_with_alpha_numeric(csv_file, column_index)

output_file = 'row_numbers.txt'

with open(output_file, 'w') as file:
    for row in rows:
        file.write(str(row) + '\n')
