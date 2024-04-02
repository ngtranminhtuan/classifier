import os
import shutil
from sklearn.model_selection import train_test_split

dataset_directory = './checkbox_state_v2/data/'  # Replace with your dataset directory

# Specify the directories for the train and val sets
train_directory = os.path.join(dataset_directory, 'train')
val_directory = os.path.join(dataset_directory, 'val')

# Create the train and val directories if they don't exist
os.makedirs(train_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)

# Split ratios for train and validation
train_ratio = 0.8
val_ratio = 0.2

# Ensure the split ratios sum up to 1, using rounding for floating-point precision
assert round(train_ratio + val_ratio, 10) == 1, "Split ratios must sum up to 1"

# Process each class directory ('checked', 'unchecked', 'other')
for class_name in ['checked', 'unchecked', 'other']:
    # Path to the class directory
    class_dir = os.path.join(dataset_directory, class_name)
    
    # Get a list of all files in the class directory
    all_files = [file for file in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, file))]

    # Split the files into train and val sets
    train_files, val_files = train_test_split(all_files, train_size=train_ratio, random_state=42)

    # Function to copy the files to the respective train or val directories
    def copy_files(files, dest_dir):
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
        for file in files:
            src_path = os.path.join(class_dir, file)
            dest_path = os.path.join(dest_dir, class_name, file)
            shutil.copy(src_path, dest_path)

    # Copy files to the respective directories
    copy_files(train_files, train_directory)
    copy_files(val_files, val_directory)

print("Done splitting into train and val sets.")
