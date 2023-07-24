import csv

def add_extension_to_first_column(csv_file):
    data = []

    # Read the original CSV file
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        for row in reader:
            # Append '.jpg' to the first column value
            row[0] = f"{row[0]}.jpg"
            data.append(row)

    # Write the updated data back to the original CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write the header
        for row in data:
            writer.writerow(row)

# Example usage:
csv_file_path = "../dataset/captions.csv"
add_extension_to_first_column(csv_file_path)