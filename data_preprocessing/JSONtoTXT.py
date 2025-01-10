import os
import json

def roboflow_to_yolo(json_path, output_dir, images_dir):
    """
    Convert RoboFlow-like JSON annotations to YOLO format.

    Args:
        json_path (str): Path to the JSON file.
        output_dir (str): Directory to save YOLO `.txt` annotation files.
        images_dir (str): Directory containing the images.
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a mapping for category IDs to YOLO IDs
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    # Map images by their IDs
    images = {img['id']: img for img in data['images']}

    # Iterate over annotations and convert to YOLO format
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # [x_min, y_min, width, height]

        # Get image details
        img = images[image_id]
        img_width = img['width']
        img_height = img['height']

        # Convert bbox to YOLO format
        x_min, y_min, box_width, box_height = bbox
        x_center = (x_min + box_width / 2) / img_width
        y_center = (y_min + box_height / 2) / img_height
        width = box_width / img_width
        height = box_height / img_height

        # Get YOLO class ID
        yolo_class_id = categories.get(category_id, -1)  # Map to YOLO ID
        if yolo_class_id == -1:
            print(f"Skipping annotation with unknown category_id: {category_id}")
            continue

        # Prepare YOLO annotation line
        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        # Write to corresponding .txt file
        image_filename = img['file_name']
        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'a') as f:
            f.write(yolo_line + '\n')

    print(f"Converted JSON annotations to YOLO format in '{output_dir}'")

# Example usage
json_annotations = "26ClassObjects/valid/_annotations.coco.json"  # Path to JSON file
output_directory = "26ClassObjects/valid_txt_annotations"  # Output directory for YOLO .txt files
images_directory = "./images/"  # Path to the images directory

roboflow_to_yolo(json_annotations, output_directory, images_directory)
