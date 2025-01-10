import os
import shutil
import random

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split a dataset into training, validation, and test sets, skipping images without labels.

    Args:
        image_dir (str): Path to the directory containing image files.
        label_dir (str): Path to the directory containing annotation files.
        output_dir (str): Base directory to store the split dataset.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int): Random seed for reproducibility.
    """
    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Get list of images
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Filter out images without corresponding label files
    image_label_pairs = []
    for image in images:
        label = os.path.splitext(image)[0] + ".txt"  # Label file corresponding to the image
        if os.path.exists(os.path.join(label_dir, label)):
            image_label_pairs.append((image, label))

    # Shuffle the dataset
    random.seed(seed)
    random.shuffle(image_label_pairs)

    # Compute split indices
    total_count = len(image_label_pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_set = image_label_pairs[:train_count]
    val_set = image_label_pairs[train_count:train_count + val_count]
    test_set = image_label_pairs[train_count + val_count:]

    # Define output directories
    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    for split_name, split_data in splits.items():
        image_output = os.path.join(output_dir, f"images/{split_name}")
        label_output = os.path.join(output_dir, f"labels/{split_name}")
        os.makedirs(image_output, exist_ok=True)
        os.makedirs(label_output, exist_ok=True)

        for image, label in split_data:
            shutil.copy(os.path.join(image_dir, image), os.path.join(image_output, image))
            shutil.copy(os.path.join(label_dir, label), os.path.join(label_output, label))

    print(f"Dataset split completed. Results saved to '{output_dir}'.")

# Example usage
image_directory = "combined dataset/images/train"  # Path to the images
label_directory = "combined dataset/labels/train"  # Path to the annotations
output_directory = "split_dataset"  # Path to save the split dataset

split_dataset(image_directory, label_directory, output_directory)
