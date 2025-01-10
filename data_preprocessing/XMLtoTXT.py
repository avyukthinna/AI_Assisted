import os
import xml.etree.ElementTree as ET

def voc_to_yolo(xml_dir, output_dir, classes, img_dimensions=None):
    """
    Convert XML annotations (Pascal VOC) to YOLO format.
    
    Args:
        xml_dir (str): Path to the folder containing XML annotation files.
        output_dir (str): Path to save YOLO `.txt` annotation files.
        classes (list): List of class names in order.
        img_dimensions (tuple or None): Image dimensions (width, height). If None, dimensions are read from XML.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Read image dimensions
        if img_dimensions:
            img_width, img_height = img_dimensions
        else:
            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)

        yolo_annotations = []

        # Process each object in the XML file
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue
            
            class_id = classes.index(class_name)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO annotations to a `.txt` file
        txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(yolo_annotations))

    print(f"Converted all XML files in {xml_dir} to YOLO format in {output_dir}.")

# Example usage
xml_directory = "potholes/annotations"  # Path to the folder containing XML files
output_directory = "potholes/potholes_txt_annotations"  # Path to save the YOLO `.txt` files
class_list = ["pothole"]  # List of classes in the dataset

voc_to_yolo(xml_directory, output_directory, classes=class_list)
