"""
Script to process a CSV file: add '.jpg' extension to the first column,
copy files from a source directory, find rows with alphanumeric values
in a specified column, split data into train and test sets, create CSV files,
and merge CSV files.

Author: Shima Baniadamdizaj
Email: baniadam.shima@gmail.com
"""

import csv
import re
import shutil
import os
import random
import pandas as pd

# Global constants
TEST_SIZE = 0.2  # 20% of the data will be used for testing
IMAGE_DIR = "../dataset/images"
CAPTIONS_CSV = "../dataset/captions.csv"
TRAIN_DIR = "../dataset/train_dir"
TEST_DIR = "../dataset/test_dir"
SOURCE_DIR_COPY = '../Flicker30k/Images/'

def contains_alpha_numeric(text):
    """
    Check if a given text contains alphanumeric characters.

    Args:
        text (str): The input text.

    Returns:
        bool: True if the text contains alphanumeric characters, False otherwise.
    """
    return bool(re.search(r'[a-zA-Z0-9]', text))

def find_rows_with_alpha_numeric(csv_file, column_index):
    """
    Find row numbers in a CSV file where the specified column contains alphanumeric values.

    Args:
        csv_file (str): Path to the CSV file.
        column_index (int): Index of the column to check.

    Returns:
        list: List of row numbers containing alphanumeric values in the specified column.
    """
    rows_with_alpha_numeric = []

    with open(csv_file, 'r', encoding='utf-8') as file:
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

def write_row_numbers_to_file(row_numbers, output_file):
    """
    Write row numbers to a text file.

    Args:
        row_numbers (list): List of row numbers.
        output_file (str): Path to the output text file.
    """
    with open(output_file, 'w') as file:
        for row in row_numbers:
            file.write(str(row) + '\n')

def add_extension_to_first_column(csv_file):
    """
    Adds '.jpg' extension to the values in the first column of a CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file.

    Returns:
    None
    """
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

def copy_files_from_csv(csv_file, source_dir, destination_dir):
    """
    Copies files listed in a CSV file from a source directory to a destination directory.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - source_dir (str): Path to the source directory.
    - destination_dir (str): Path to the destination directory.

    Returns:
    None
    """
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

def split_data(image_dir, train_dir, test_dir, test_size):
    """
    Split data into train and test sets.

    Parameters:
    - image_dir (str): Path to the directory containing images.
    - train_dir (str): Path to the directory for the training set.
    - test_dir (str): Path to the directory for the test set.
    - test_size (float): Percentage of data to be used for testing.

    Returns:
    None
    """
    # Get a list of all image files in the image_dir
    image_files = [file for file in os.listdir(image_dir) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    num_images = len(image_files)
    
    # Calculate the number of images for testing and training
    num_test_images = int(num_images * test_size)
    num_train_images = num_images - num_test_images

    # Shuffle the image_files list to ensure random selection for train and test sets
    random.shuffle(image_files)

    # Copy images to the train_dir
    for i in range(num_train_images):
        src_file = os.path.join(image_dir, image_files[i])
        dst_file = os.path.join(train_dir, image_files[i])
        shutil.copyfile(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")

    # Copy images to the test_dir
    for i in range(num_train_images, num_images):
        src_file = os.path.join(image_dir, image_files[i])
        dst_file = os.path.join(test_dir, image_files[i])
        shutil.copyfile(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")

def create_image_list_csv(image_dir, csv_file):
    """
    Create a CSV file with a list of image names in a directory.

    Parameters:
    - image_dir (str): Path to the directory containing images.
    - csv_file (str): Path to the CSV file to be created.

    Returns:
    None
    """
    # Get a list of all files in the image directory
    image_files = os.listdir(image_dir)

    # Filter only the files with image extensions (you can add more extensions if needed)
    image_files = [file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Write the image names to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name'])  # Write the header
        for file_name in image_files:
            writer.writerow([file_name])

def read_csv(csv_file):
    """
    Read a CSV file and return the data.

    Parameters:
    - csv_file (str): Path to the CSV file.

    Returns:
    list: List of rows from the CSV file.
    """
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        for row in reader:
            data.append(row)
    return data

def write_csv(csv_file, data):
    """
    Write data to a CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - data (list): Data to be written to the CSV file.

    Returns:
    None
    """
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'second_column'])  # Write the header
        for row in data:
            writer.writerow(row)

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merge two CSV files based on the first column and update the second column.

    Parameters:
    - file1_path (str): Path to the first CSV file.
    - file2_path (str): Path to the second CSV file.
    - output_path (str): Path to the merged CSV file.

    Returns:
    None
    """
    # Read both CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge the DataFrames on the first column
    merged_df = pd.merge(df1, df2, on=df1.columns[0], how='left')

    # Update the second column of the first DataFrame with matching values from the second DataFrame
    merged_df[df1.columns[1]] = merged_df[df1.columns[1]]

    # Drop the duplicated column from the merge
    merged_df.drop(columns=[df1.columns[1]], inplace=True)

    # Write the merged DataFrame back to the output file
    merged_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage for adding extensions to the first column
    add_extension_to_first_column(CAPTIONS_CSV)

    # Example usage for copying files from a CSV

    copy_files_from_csv(CAPTIONS_CSV, SOURCE_DIR_COPY, IMAGE_DIR)

    # Example usage for finding rows with alphanumeric values
    csv_file_path_find = '../dataset/captions_all.csv'
    column_index_find = 2
    output_file_find = 'row_numbers.txt'

    rows_find = find_rows_with_alpha_numeric(csv_file_path_find, column_index_find)

    # Write row numbers to the output file
    write_row_numbers_to_file(rows_find, output_file_find)

    # Example usage for splitting data into train and test sets
    split_data(IMAGE_DIR, TRAIN_DIR, TEST_DIR, TEST_SIZE)
    
    # Example usage for creating CSV files with image names
    csv_file_train = "../dataset/train_captions.csv"
    csv_file_test = "../dataset/test_captions.csv"

    create_image_list_csv(TRAIN_DIR, csv_file_train)
    create_image_list_csv(TEST_DIR, csv_file_test)
    print("CSV file with image names has been created.")

    # ATTENTION: should open csv file and add one column
    merge_csv_files(csv_file_train, CAPTIONS_CSV, csv_file_train)
    merge_csv_files(csv_file_test, CAPTIONS_CSV, csv_file_test)
    print("Second column values copied and updated in the original CSV file.")
