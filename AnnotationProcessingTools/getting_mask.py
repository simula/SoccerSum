import os
import json
import requests
import shutil  # Import shutil for creating zip files

# Function to create a directory if it doesn't exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Function to download and save an image from a given URL
def download_and_save_image(image_url, image_path, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(image_url, headers=headers)
    if response.status_code == 200:
        with open(image_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download image from {image_url}. Status code: {response.status_code}")

# Function to process the NDJSON file and download images
def process_ndjson_file(ndjson_path, output_folder, api_key):
    create_directory(output_folder)  # Ensure the output directory exists
    with open(ndjson_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Navigate through the nested JSON to find each "mask"
            projects = data.get("projects", {})
            for project_id, project_data in projects.items():
                labels = project_data.get("labels", [])
                for label in labels:
                    annotations = label.get("annotations", {})
                    objects = annotations.get("objects", [])
                    for obj in objects:
                        if obj.get("annotation_kind") == "ImageSegmentationMask":
                            mask = obj.get("mask", {})
                            image_url = mask.get("url")
                            if image_url:  # Only proceed if there's a URL to download
                                external_id = data["data_row"]["external_id"]
                                feature_id = obj.get("name", "no_id")
                                image_filename = f"{external_id}_{feature_id}.png"
                                image_path = os.path.join(output_folder, image_filename)
                                download_and_save_image(image_url, image_path, api_key)

# Function to zip the output directory
def zip_output_directory(output_folder, zip_name):
    shutil.make_archive(zip_name, 'zip', output_folder)


# Define the path to your NDJSON file and the output folder
ndjson_file_path = '/content/export-result_(3).ndjson'
output_directory = 'path_to_your_output_directory3'

# Insert your API key here
api_key = 'YOUR_API_KEY'

# Call the function to process the NDJSON file
process_ndjson_file(ndjson_file_path, output_directory, api_key)


# Once the downloading is done, zip the output folder
zip_output_directory(output_directory, 'segmentation_masks')


print("Downloading of segmentation masks is complete.")
print("Downloading of segmentation masks is complete. The output directory is now zipped.")
