import csv
import shutil
import os

def copy_files_from_csv(csv_file, source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        rows.pop(0)  # Remove the first row

        for row in rows:
            file_name = row[0]  # Assuming the file name is in the first column

            # Add .jpg extension to the file name
            file_name_with_extension = file_name + '.jpg'

            # Construct the source and destination paths
            source_path = os.path.join(source_dir, file_name_with_extension)
            destination_path = os.path.join(destination_dir, file_name_with_extension)

            # Copy the file
            shutil.copy2(source_path, destination_path)
            print(f"Copied {file_name_with_extension} to {destination_dir}")

# Example usage
csv_file = '../dataset/captions.csv'
source_dir = '../Flicker30k/Images/'
destination_dir = '../dataset/images/'

copy_files_from_csv(csv_file, source_dir, destination_dir)
