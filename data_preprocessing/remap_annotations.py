import os

def remap_class_ids(txt_dir, master_class_list):
    """
    Remap class IDs in YOLO .txt annotation files based on a master class list.

    Args:
        txt_dir (str): Path to the directory containing YOLO .txt annotation files.
        master_class_list (list): List of class names in the desired order.
    """
    # Map old IDs to new IDs based on the master class list
    id_mapping = {}
    for i, class_name in enumerate(master_class_list):
        id_mapping[class_name] = i
    print(id_mapping)
    # Process each .txt file in the directory
    for txt_file in os.listdir(txt_dir):
        if not txt_file.endswith(".txt"):
            continue

        txt_path = os.path.join(txt_dir, txt_file)
        remapped_lines = []

        # Read the original annotation file
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Skipping invalid line in {txt_file}: {line}")
                    continue
                
                old_class_id = int(parts[0])
                bbox_data = parts[1:]  # x_center, y_center, width, height

                # Find the corresponding new class ID
                new_class_id = id_mapping.get("text_region", -1)
                print(new_class_id)
                if new_class_id == -1:
                    print(f"Skipping unknown class ID {old_class_id} in {txt_file}")
                    continue
                
    
                # Create the remapped line
                remapped_line = f"{new_class_id} {' '.join(bbox_data)}"
                remapped_lines.append(remapped_line)

        # Overwrite the file with remapped annotations
        with open(txt_path, "w") as f:
            f.write("\n".join(remapped_lines))
        print(f"Remapped class IDs in {txt_file}")

# Example usage
txt_directory = "COCO-text/annotations"  # Path to the directory containing YOLO .txt files
master_classes = [
    "cars-bikes-people",
    "Bus",
    "Bushes",
    "Person",
    "Truck",
    "backpack",
    "bench",
    "bicycle",
    "boat",
    "branch",
    "car",
    "chair",
    "clock",
    "crosswalk",
    "door",
    "elevator",
    "fire_hydrant",
    "green_light",
    "gun",
    "handbag",
    "motorcycle",
    "person",
    "pothole",
    "rat",
    "red_light",
    "scooter",
    "sheep",
    "stairs",
    "stop_sign",
    "suitcase",
    "traffic light",
    "traffic_cone",
    "train",
    "tree",
    "truck",
    "umbrella",
    "yellow_light",
    "text_region"
]

remap_class_ids(txt_directory, master_classes)
