import os
import random
import shutil
import csv
import pandas as pd


TEST_SIZE = 0.2  # 20% of the data will be used for testing
IMAGE_DIR = "../dataset/images"
TRAIN_DIR = "../dataset/train_dir"
TEST_DIR = "../dataset/test_dir"
CAPTIONS_CSV = "../dataset/captions.csv"

def split_data(image_dir, train_dir, test_dir, test_size):
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
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        for row in reader:
            data.append(row)
    return data

def write_csv(csv_file, data):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'second_column'])  # Write the header
        for row in data:
            writer.writerow(row)


def merge_csv_files(file1_path, file2_path, output_path):
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
    # Create the directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Split and copy the images to train and test directories
    split_data(IMAGE_DIR, TRAIN_DIR, TEST_DIR, TEST_SIZE)
    csv_file_train = "../dataset/train_captions.csv"
    csv_file_test = "../dataset/test_captions.csv"

    create_image_list_csv(TRAIN_DIR, csv_file_train)
    create_image_list_csv(TEST_DIR, csv_file_test)
    print("CSV file with image names has been created.")

    # ATTENTION: should open csv file adn add one column
    merge_csv_files(csv_file_train, CAPTIONS_CSV, csv_file_train)
    merge_csv_files(csv_file_test, CAPTIONS_CSV, csv_file_test)
    print("Second column values copied and updated in the original CSV file.")
