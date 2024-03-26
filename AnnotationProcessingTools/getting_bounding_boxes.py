import json
import os
import zipfile

def convert_to_yolo(bbox, dw, dh):
    x = (bbox['left'] + bbox['width'] / 2.0) / dw
    y = (bbox['top'] + bbox['height'] / 2.0) / dh
    w = bbox['width'] / dw
    h = bbox['height'] / dh
    return f"{x} {y} {w} {h}"

# Define your class mapping here
class_mapping = {
    'Player': 0,
    'Goalkeeper': 1,
    'Referee': 2,
    'Ball': 3,
    'Logo': 4,
    'Penalty Mark': 5,
    'Corner Flagpost': 6,
    'Goal Net': 7
}

# Set your paths here
json_file_path = '/content/export-result_part3.ndjson'  # Replace with your JSON file path
output_directory = '_annotation_Part3'  # Replace with your desired output directory

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read the JSON file
with open(json_file_path, 'r') as file:
    annotations_data = [json.loads(line) for line in file]

# Process each annotation in the JSON file
for data_row in annotations_data:
    external_id = data_row["data_row"]["external_id"]
    image_width = data_row["media_attributes"]["width"]
    image_height = data_row["media_attributes"]["height"]

    # Initialize an empty list to hold YOLO formatted annotations for this image
    yolo_annotations = []

    # Check if there are any annotations
    annotations = data_row["projects"]["clomy11oh06d8082n75zv0oy2"]["labels"][0]["annotations"].get("objects", [])
    for annotation in annotations:
        if 'bounding_box' in annotation and annotation.get('annotation_kind') == 'ImageBoundingBox':
            class_name = annotation["name"]
            # Use the class mapping to convert class names to indices
            class_index = class_mapping.get(class_name)
            if class_index is not None:
                yolo_bbox = convert_to_yolo(annotation["bounding_box"], image_width, image_height)
                yolo_annotations.append(f"{class_index} {yolo_bbox}")
            else:
                print(f"Warning: Class name '{class_name}' not found in class mapping for image {external_id}.")

    # Write to the corresponding .txt file for each image
    yolo_filename = os.path.join(output_directory, os.path.splitext(external_id)[0] + '.txt')
    with open(yolo_filename, 'w') as yolo_file:
        for item in yolo_annotations:
            yolo_file.write(f"{item}\n")

# Print completion message
print("YOLO annotation files have been created.")


# After the loop that writes all the .txt files, create a ZIP file
zip_filename = "annotations_part3.zip"
zip_filepath = os.path.join(output_directory, zip_filename)

with zipfile.ZipFile(zip_filepath, 'w') as zipf:
    # Walk the directory and add all .txt files to the zip
    for root, _, files in os.walk(output_directory):
        for file in files:
            if file.endswith(".txt"):
                zipf.write(os.path.join(root, file), file)

# Print completion message
print(f"YOLO annotation files have been created and zipped into {zip_filename}.")
