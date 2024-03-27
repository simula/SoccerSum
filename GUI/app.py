from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
from ultralytics import YOLO
import json
import torch
import torchvision
import time
import subprocess
import os
from collections import defaultdict
import cv2
import numpy as np
import urllib.parse
import requests
import openai
import whisper
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
import re

# Current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#object_detected_file_path = f'/home/mehdihou/D1/02_Thesis/objects_{timestamp}.json'


has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)
# Once https://github.com/pytorch/pytorch/issues/77818 is resolved we can try again to run on MPS
has_mps = False
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"


class Frame_Type:
    def __init__(self, category, subcategory, start_frame, end_frame):
        
        self.category = category
        self.subcategory = subcategory
        self.start_frame = start_frame
        self.end_frame = end_frame

    def dict(self):
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame
        }



def download_and_convert_m3u8(m3u8_url, output_mp4_path):
    # Determine the ffmpeg command based on file type
    if m3u8_url.endswith('.m3u8'):
        ffmpeg_command = ['ffmpeg', '-i', m3u8_url, '-c', 'copy', '-bsf:a', 'aac_adtstoasc', output_mp4_path]
    elif m3u8_url.endswith('.mp4'):
        ffmpeg_command = ['ffmpeg', '-i', m3u8_url, '-c', 'copy', output_mp4_path]
    else:
        raise ValueError("Unsupported file format. Please provide a '.m3u8' or '.mp4' file.")

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during download/conversion: {e}")
        return None

    return output_mp4_path

def process_video(video_path, timestamp):
    # Simplify or sanitize the filename
    parsed_url = urllib.parse.urlparse(video_path)
    base_filename = os.path.basename(parsed_url.path)
    simple_filename = os.path.splitext(base_filename)[0] + f'_{timestamp}.mp4'

    # Specify a directory for saving the output file
    output_directory = '/home/mehdi/03_GUI/uploads'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_mp4_path = os.path.join(output_directory, simple_filename)

    if video_path.endswith('.m3u8') or video_path.startswith('http://') or video_path.startswith('https://'):
        video_path = download_and_convert_m3u8(video_path, output_mp4_path)

    return video_path




def reduce_fps(input_video_path, method='set_fps', target_fps=1, frame_selection_rate=25):
    """
    Reduce the frame rate of a video either uniformly or by setting a target FPS, 
    and save it in a 'reduced_fps' subdirectory.

    :param input_video_path: Path to the input video.
    :param method: The method of reducing FPS ('uniform' or 'set_fps').
    :param target_fps: Desired frames per second (used for 'set_fps' method).
    :return: Path where the reduced FPS video was saved, or None if an error occurred.
    """
    base_dir = os.path.dirname(input_video_path)
    base_filename = os.path.basename(input_video_path)
    output_dir = os.path.join(base_dir, 'reduced_fps')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reduced_fps_video_path = os.path.join(output_dir, f'reduced_fps_{base_filename}')

    if method == 'uniform':
        # Use frame_selection_rate in ffmpeg command
        ffmpeg_command = [
            'ffmpeg', '-i', input_video_path, 
            '-vf', f'select=not(mod(n\,{frame_selection_rate}))', 
            '-vsync', 'vfr', reduced_fps_video_path
        ]
    elif method == 'set_fps':
        # Set target FPS
        ffmpeg_command = ['ffmpeg', '-i', input_video_path, '-filter:v', f'fps=fps={target_fps}', reduced_fps_video_path]
    else:
        raise ValueError("Invalid method. Choose 'uniform' or 'set_fps'.")

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while reducing FPS: {e}")
        return None

    return reduced_fps_video_path



    
def load_shot_prediction_model():
    # load shot type classification model, training available in https://github.com/sssabet/Shot_Type_Classification
    if device == 'cpu':
        shot_model = torch.load('/home/mehdi/03_GUI/models/shot_type_classification.pt', map_location =torch.device('cpu'))
    else:
        shot_model = torch.load('/home/mehdi/03_GUI/models/shot_type_classification.pt')
    return shot_model



shot_model = load_shot_prediction_model()


# Function that i-recives a frame(128,128) and model 2-does the nessaccery transformations, 3-predicts the types of shot and 4-returns the predicted label of frame type shot
# Works fast on cuda but takes some time for cpu, lower image sizes speeds up the proccess with a lower accuracy
def predict_shot(model, frame):
    IMAGE_SIZE = (128,128)        
    data_transformation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Resize(IMAGE_SIZE),torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    images = data_transformation(frame)
    with torch.no_grad():
        pred = model(images.view([1, 3, 128, 128]).to(device))
        _ , predictes = torch.max(pred,1)

    if predictes.item() == 0:
        predicted_type = 'CS'
    elif predictes.item() == 1:
        predicted_type = 'ECS'
    elif predictes.item() == 2:
        predicted_type = 'FS'
    elif predictes.item() == 3:
        predicted_type = 'LS'
    else:
        predicted_type = 'MS'

    return predicted_type


def predict_frame_type(video_path, timestamp,selected_shots, model=shot_model) -> [Frame_Type]:
    frame_type = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_n = 0
    start = 1
    predicte = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % fps == 0:  # TO reduce the computation run the model once for each half a second
            old_predicte = predicte
            predicte = predict_shot(model,frame)
            
            if old_predicte != predicte or frame_n > cap.get(cv2.CAP_PROP_FRAME_COUNT)-fps:  # if new frame type or the last frame 
                
                frame_type.append(Frame_Type('frame_type', predicte, start_frame= start, end_frame=frame_n-1))
      
                start = frame_n
    

    # Saving transitions into a JSON file
    filename = f'/home/mehdi/03_GUI/static/generated_files/frame_type{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump([ft.dict() for ft in frame_type], f)

    if os.path.isfile(filename):
        print(f"{filename} has been created and frame_types have been written successfully.")
    else:
        print(f"Failed to create {filename}.")



    # Filter for 'LS' subcategory frames
    ls_frames = [ft for ft in frame_type if ft.subcategory in selected_shots]

    # Open the original video
    cap = cv2.VideoCapture(video_path)

    
    
    # Define the output file path
    output_file_name = f'output_ls_{timestamp}.mp4'
    output_path = os.path.join('/home/mehdi/03_GUI/static/generated_files', output_file_name)
    
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'

    # Create VideoWriter with the specified path
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


    print(f"Video will be saved to: {output_path}")

    # Read through the video and write frames that belong to 'LS' subcategory
    frame_n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        if any(ft.start_frame <= frame_n <= ft.end_frame for ft in ls_frames):
            out.write(frame)

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()


    return frame_type, output_path




#Starting with Detected Objects

class DetectedObjects:
    def __init__(self, category, confidence, x1, y1, x2, y2, frame_number):
        self.category = category
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return{
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }


class SegmentedObject:
    def __init__(self, category, confidence, x1, y1, x2, y2, frame_number):
        self.category = category
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return{
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }
    

def detect_yolo(output_path, selected_YDM):
    class_names = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
    ]

    
    if selected_YDM == 'YD1':
        model = YOLO('/home/mehdi/03_GUI/models/detection_best.pt')
        class_names = ['Player', 'Goalkeeper', 'Referee', 'Ball', 'Logo', 'Penalty Mark', 'Corner Falgpost', 'Goal Net']
    
    elif selected_YDM == 'YD2':
        model = YOLO('/home/mehdi/03_GUI/models/Detection/01- nano/yolov8n.pt')
    elif selected_YDM == 'YD3':
        model = YOLO('/home/mehdi/03_GUI/models/Detection/02 - small/yolov8s.pt')
    elif selected_YDM == 'YD4':
        model = YOLO('/home/mehdi/03_GUI/models/Detection/03 - medium/yolov8m.pt')
    elif selected_YDM == 'YD5':
        model = YOLO('/home/mehdi/03_GUI/models/Detection/04 - large/yolov8l.pt')
    elif selected_YDM == 'YD6':
        model = YOLO('/home/mehdi/03_GUI/models/Detection/05 - xlarge/yolov8x.pt')
    elif selected_YDM == 'YD7':
        model = YOLO('/home/mehdi/03_GUI/models/yolov8m-football.pt')

    #Running the Inference on the video
    results = model.predict(output_path, stream=True)

    #Creating a List to store the detected Objects
    detected_objects = []

    # Dictionary to keep track of object counts
    object_counts = {}

    for frame_n, result in enumerate(results):
        # Moving the object to the CPU memory
        result = result.cpu()
        
        # The Boxes object containing the detection bounding boxes
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf)
            clazz = int(box.cls)

            # Adding the detected object
            detected_objects.append(DetectedObjects(class_names[clazz], conf, x1, y1, x2, y2, frame_n))

            # Counting the objects
            class_name = class_names[clazz]
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1

    # Return the total number of frames processed
    total_frames = frame_n + 1
    print(f"The Total Number of Frames Processed is {total_frames}")
    return detected_objects, object_counts, total_frames


def segment_yolo(output_path, selected_YSM):
    class_names = [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush'
    ]
    
    if selected_YSM == 'YS1':
        model = YOLO("/home/mehdi/03_GUI/models/Seg_best.pt")
        class_names = ['PenaltyBox', 'GoalBox']

    elif selected_YSM == 'YS2':
        model = YOLO('/home/mehdi/03_GUI/models/Segmentation/01 - nano/yolov8n-seg.pt')
    elif selected_YSM == 'YS3':
        model = YOLO('/home/mehdi/03_GUI/models/Segmentation/02 - small/yolov8s-seg.pt')
    elif selected_YSM == 'YS4':
        model = YOLO('/home/mehdi/03_GUI/models/Segmentation/03 - medium/yolov8m-seg.pt')
    elif selected_YSM == 'YS5':
        model = YOLO('/home/mehdi/03_GUI/models/Segmentation/04 - large/yolov8l-seg.pt')
    elif selected_YSM == 'YS6':
        model = YOLO('/home/mehdi/03_GUI/models/Segmentation/05 - xlarge/yolov8x-seg.pt')




    results = model.predict(output_path, stream=True)

    segmented_objects = []

    # Dictionary to keep track of object counts
    segment_counts = {}


    for frame_n, result in enumerate(results):

        result = result.cpu()

        #the Boxes object containg the detection bouding boxes
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf)

            clazz = int(box.cls)

            segmented_objects.append(SegmentedObject(class_names[clazz], conf, x1, y1, x2, y2, frame_n))

            # Counting the objects
            class_name = class_names[clazz]
            if class_name in segment_counts:
                segment_counts[class_name] += 1
            else:
                segment_counts[class_name] = 1

    return segmented_objects, segment_counts


class TrackedObject:
    def __init__(self, category, track_id, x1, y1, x2, y2, frame_number):
        self.category = category
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return {
            "category": self.category,
            "track_id": self.track_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }

def track_yolo(output_path, selected_YTO, timestamp):
    model = YOLO('/home/mehdi/03_GUI/models/detection_best.pt')
    cap = cv2.VideoCapture(output_path)

    if selected_YTO == 'YT1':
        selected_YTO = 'Goalkeeper'
    elif selected_YTO == 'YT2':
        selected_YTO = 'Player'
    elif selected_YTO == 'YT3':
        selected_YTO = 'Ball'
    
    tracked_objects = []
    class_names = ['Player', 'Goalkeeper', 'Referee', 'Ball', 'Logo', 'Penalty Mark', 'Corner Flagpost', 'Goal Net']  # Define your class names here
    print(f'We are tracking {selected_YTO}')
    goalkeeper_class_id = class_names.index(selected_YTO)

    frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)

            if results and results[0] and results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id
                class_ids = results[0].boxes.cls

                if track_ids is not None and class_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                    class_ids = class_ids.int().cpu().tolist()

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        if class_id == goalkeeper_class_id:  # Filter for Goalkeepers
                            x, y, w, h = box
                            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                            tracked_objects.append(TrackedObject(selected_YTO, track_id, x1, y1, x2, y2, frame_number))
            
            frame_number += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Coordinates for plotting
    x_coords = []
    y_coords = []
    frame_numbers = []

    for tracked_obj in tracked_objects:
        if tracked_obj.category == selected_YTO:
            # Calculate the center of the bounding box
            center_x = (tracked_obj.x1 + tracked_obj.x2) / 2
            center_y = (tracked_obj.y1 + tracked_obj.y2) / 2

            x_coords.append(center_x)
            y_coords.append(center_y)
            frame_numbers.append(tracked_obj.frame_number)

    print(f"X coordinates: {x_coords} \ Year coordinates: {y_coords} \ Frame numbers: {frame_numbers}")

    # Creating the 3D plot with a transparent background
    fig = plt.figure()
    fig.patch.set_alpha(0.0)  # Set the outer background to transparent
    ax = fig.add_subplot(111, projection='3d', facecolor=(0,0,0,0))  # Set the axes background to transparent

    # Swapping the role of frame_numbers with one of the other coordinates
    # Here, frame_numbers is now on the Z-axis
    ax.scatter(x_coords, y_coords, frame_numbers, c='r', marker='o')
    ax.plot(x_coords, y_coords, frame_numbers, color='b')

    # Update the labels to reflect the new axis arrangement
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Frame Number')
    ax.set_title(f'Tracking Movement of {selected_YTO}')

    # Save the plot as an image
    image_filename = f"tracking_{timestamp}.png"
    tracking_path = os.path.join('static', 'animations', image_filename)

    plt.savefig(tracking_path)


    return tracked_objects, tracking_path


#SmartCobo - Trying to Group the elements in the text file

def reduce_floats_in_item(item):
    """ Recursively reduce float numbers in a JSON item (dict or list) """
    if isinstance(item, dict):
        return {key: reduce_floats_in_item(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [reduce_floats_in_item(elem) for elem in item]
    elif isinstance(item, float):
        return round(item, 1)
    return item

def format_and_write_txt(data, txt_path):
    """ Format and group objects by frame number, then write to a text file """
    with open(txt_path, 'w') as file:
        frame_data = {}
        # Grouping objects by frame_number
        for list_name, list_items in data.items():
            if isinstance(list_items, list):
                for obj in list_items:
                    frame = obj.get("frame_number", "unknown")
                    frame_data.setdefault(frame, {}).setdefault(list_name, []).append(obj)

        for frame, objects in frame_data.items():
            file.write(f"frame_number: {frame} {{\n")
            for list_name, items in objects.items():
                file.write(f"  {list_name}:\n")
                for obj in items:
                    line = " ".join([f"{obj.get(key, '')}" for key in obj if key != "frame_number"])
                    file.write(f"    {line}\n")
            file.write("}\n\n")

def reduce_floats_and_convert_to_txt(input_path, output_txt_path):
    """ Process JSON file, reduce floats, and convert to text format """
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)

        modified_data = reduce_floats_in_item(data)
        format_and_write_txt(modified_data, output_txt_path)

        return "TXT file generated successfully."
    except Exception as e:
        return f"Error processing file: {e}"



def filter_frames(file_path, min_detections):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filtered_data = []
    frame_info = []
    detection_count = 0
    in_detection_block = False
    in_metadata_section = True

    for line in lines:
        if in_metadata_section and 'frame_number' not in line:
            filtered_data.append(line)
        else:
            in_metadata_section = False
            if 'frame_number' in line:
                # Check if previous frame should be kept
                if detection_count >= min_detections:
                    filtered_data.extend(frame_info)
                frame_info = [line]
                detection_count = 0
                in_detection_block = False
            elif 'detected_objects:' in line:
                frame_info.append(line)
                in_detection_block = True
            elif in_detection_block and line.strip() and not (line.strip().startswith('}') or line.strip().startswith('segmented_objects:')):
                frame_info.append(line)
                detection_count += 1
            else:
                if in_detection_block and (line.strip().startswith('}') or line.strip().startswith('segmented_objects:')):
                    in_detection_block = False
                frame_info.append(line)

    # Check the last frame in the file
    if detection_count >= min_detections:
        filtered_data.extend(frame_info)

    # Writing back to the file or return filtered data
    with open(file_path, 'w') as file:
        file.writelines(filtered_data)



def chat_with_openai(openai_api_key, text_file_path, temperature=0.7, max_tokens=1000, top_p=1, presence_penalty=0, frequency_penalty=0, seed=None):
    openai.api_key = openai_api_key

    # Read the contents of the text file
    with open(text_file_path, 'r') as file:
        file_contents = file.read()

    # Initial prompt
    initial_prompt = """Compose a series of tweets. Start the tweets with "Tweet:", each with a maximum of 280 characters, narrating the progression of a soccer goal. Use the provided OBJECT DETECTION, PITCH SEGMENTATION, OBJECT TRACKING data from key frames to detail the attack's initiation, build-up, and final goal moment. USE AUDIO TRANSCRIPTS of NARRATION.Include the position of all the detected objects in compare to each other. In which part of the pitch ball and other player are moving and etc. Each tweet should capture a different stage of the play, highlighting player movements, ball trajectory, and field positions. The narrative should start with the attack's setup, move through the key passes and strategic plays, climax with the goal, and conclude with the immediate aftermath of the event. Ensure the language is vivid and engaging, encapsulating the excitement of the crowd and the skill on display. Do not reference frame numbers or technical metadata. Instead, translate the data into a dynamic and flowing story that brings the goal to life for your audience."""

    # Prepare conversation history
    conversation_history = [{"role": "system", "content": initial_prompt + "\n" + file_contents}]

    try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed
            )
            print(response)

            # Extract the AI's response
            ai_response = response.choices[0].message.content if response.choices else "No response received."
            return ai_response, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)

###############################################################################################################
###############################################################################################################
#Process Video Transcript with Whisper OpenAI
            
def extract_audio_from_video(video_url, output_audio_path, timestamp):
    # Filename for the downloaded video with timestamp
    downloaded_video_filename = f"downloaded_video_{timestamp}.mp4"

    # Download the video
    download_command = ["ffmpeg", "-i", video_url, "-c", "copy", downloaded_video_filename]
    subprocess.run(download_command, check=True)

    # Extract audio from the downloaded video
    extract_command = ["ffmpeg", "-i", downloaded_video_filename, "-vn", "-acodec", "libmp3lame", output_audio_path]
    subprocess.run(extract_command, check=True)


def calculate_similarity(segment1, segment2):
    vectorizer = TfidfVectorizer().fit([segment1, segment2])
    tfidf_matrix = vectorizer.transform([segment1, segment2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim[0][0]

def remove_overlap(transcripts, overlap_threshold=0.5, sample_length=100):
    cleaned_transcripts = [transcripts[0]]
    for i in range(1, len(transcripts)):
        prev_end = cleaned_transcripts[-1][-sample_length:]
        curr_start = transcripts[i][:sample_length]

        if calculate_similarity(prev_end, curr_start) > overlap_threshold:
            overlap_point = curr_start in prev_end
            cleaned_transcripts.append(transcripts[i][overlap_point:])
        else:
            cleaned_transcripts.append(transcripts[i])

    return cleaned_transcripts

def process_video_transcript(video_url, timestamp, model_name):
    # Path for the extracted audio file
    extracted_audio_path = f"/home/mehdi/03_GUI/static/generated_files/extracted_audio_{timestamp}.mp3"

    # Extract audio from the video
    extract_audio_from_video(video_url, extracted_audio_path, timestamp)

    model = whisper.load_model(model_name)
    sample_rate = 16000

    # Load the entire audio file
    audio = whisper.load_audio(extracted_audio_path)
    audio_length_seconds = len(audio) / sample_rate
    segment_size = 30  # seconds
    overlap = 5  # seconds for overlap

    segment_starts = [i * (segment_size - overlap) * sample_rate for i in range(math.ceil(audio_length_seconds / (segment_size - overlap)))]
    transcripts = []

    for i, start in enumerate(segment_starts):
        end = min(start + segment_size * sample_rate, len(audio))
        segment_audio = audio[start:end]

        if len(segment_audio) < segment_size * sample_rate:
            padding = np.zeros((segment_size * sample_rate - len(segment_audio),))
            segment_audio = np.concatenate((segment_audio, padding))

        segment_audio = torch.FloatTensor(segment_audio)
        mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)

        if i == 0:
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions()
        try:
            result = whisper.decode(model, mel, options)
            transcripts.append(result.text)
        except Exception as e:
            print(f"Error processing segment {i+1}: {e}")

    cleaned_transcripts = remove_overlap(transcripts)

    final_transcript = " ".join(cleaned_transcripts)
    return final_transcript, extracted_audio_path
    
###############################################################################################
###############################################################################################
# Process the Root Mean Error of the Audio of the video clip to determine the Loudiness in each frame.

def RME_audio(audio_path, timestamp, frame_length, hop_length):

    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_path)

    # Calculate RMS
    rms = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=256)

    # Flatten the RMS values to a 1D array
    rms_values = rms.flatten()
    
    # Adjust NumPy print options to display all elements
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
    # Convert the array to a string
    rms_values_str = np.array2string(rms_values, separator=', ')

    # Calculate the duration of each RMS frame
    frame_length = len(audio_data) / len(rms_values)

    # Create a time axis for the RMS values
    time = np.arange(len(rms_values)) * (frame_length / sample_rate)

    # Setting up the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0,0,0,0)) 
    ax.set_title('RMS Values Over Time')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('RMS Value')
    ax.grid(True)

    # Plot the RMS values
    ax.plot(time, rms_values, lw=3)

    # Set the x-axis and y-axis limits
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(min(rms_values), max(rms_values))

    # Save the plot as an image
    image_filename = f"rms_plot_{timestamp}.png"
    animation_file_path = os.path.join('static', 'animations', image_filename)
    fig.savefig(animation_file_path)

    plt.close(fig)  # Close the plot to free up memory

    return rms_values_str, animation_file_path



#############################################################################################################
# Extracting Frames from the Video

def extract_frames(video_path, timestamp, frame_rate=1, output_folder='static/extracted_frames'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    frame_paths = []

    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % int(fps/frame_rate) == 0:
            frame_path = os.path.join(output_folder, f"frame_{timestamp}_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            count += 1
    cap.release()
    return frame_paths, video_path









app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
progress = 0

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Current timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    global progress  

    if request.method == 'POST':
        progress = 10

        video_path = None

        # Get file path from upload or direct URL
        video_file = request.files.get('video_file')
        video_url = request.form.get('video_url')

        if video_file and video_file.filename != '':
            # Extract the base name and extension
            base_name, extension = os.path.splitext(video_file.filename)

            # Create a new filename with the timestamp
            new_filename = f"{base_name}_{timestamp}{extension}"

            # Construct the full path for the new file
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            # Save the file with the new name
            video_file.save(video_path)

        elif video_url:
            video_path = video_url

        progress = 20
    

        # Retrieve checkbox states
        enable_transcript = 'enableTranscript' in request.form
        enable_rme = 'enableRME' in request.form and enable_transcript
        
        processed_video_path = process_video(video_path, timestamp)

        # Check if user opted for manual frame selection
        manual_frame_selection = 'manual_frame_selection' in request.form
        
        if manual_frame_selection:
            frame_paths, processed_video_path = extract_frames(processed_video_path, timestamp, frame_rate=1)

            # Incorporate these results into a response
            response_frames = {
                "frame_paths": frame_paths,
                "processed_video_path": processed_video_path
            }
            return jsonify(response_frames)

        else:
            progress = 30
            #Fetchoing the Audio Transcript of the Video
            final_transcript, audio_path = (None, None)
            rms_values_str = None

            if enable_transcript:
                # Process video transcript
                model_name = request.form.get('whisperModel', 'base')
                print(f'{model_name} is  selecteeed:') #YD6
                final_transcript, audio_path = process_video_transcript(video_path, timestamp, model_name)
                print(f"The Transcript = {final_transcript} \n From the {video_path}")
            
            progress = 40        

            if enable_rme:
                # Analyze the RME of the Audio
                frame_length= request.form.get('frameLength', '1024')
                print(f'{frame_length} is  selecteeed:') #YD6

                hop_length= request.form.get('hopLength', '256')
                print(f'{hop_length} is  selecteeed:') #YD6
                
                start_time = time.time()
                rms_values_str, animation_file_path = RME_audio(audio_path, timestamp, frame_length, hop_length)
                
                end_time = time.time()
                execution_time = end_time - start_time

                librosa_exe_time = f"time_result_{timestamp}.json"

                json_file_path = os.path.join('static', 'exe_time', librosa_exe_time)

                time_result = {
                    "Start Time": start_time,
                    "End Time": end_time,
                    "execution Time": execution_time
                }
                
                with open(json_file_path, 'w') as file:
                    json.dump(time_result, file, indent=4)

                
                print(f'The RME value of the the Audio of the video clip is = \n {rms_values_str}')
            

            if processed_video_path:

                # Retrieve FPS reduction settings
                fps_method = request.form.get('fpsMethod', 'set_fps')
                target_fps = int(request.form.get('targetFps', 1))
                frame_selection_rate = int(request.form.get('frameSelectionRate', 25))

                # Reduce the FPS of the processed video
                reduced_fps_path = reduce_fps(processed_video_path, method=fps_method, target_fps=target_fps, frame_selection_rate=frame_selection_rate)
                progress = 50


                if reduced_fps_path:
                    # Unpack the returned values from predict_frame_type

                    selected_shots = request.form.getlist('shotTypeModel')
                    print(f"Shottypes are {selected_shots}")
                    
                    frame_type, output_path = predict_frame_type(reduced_fps_path, timestamp, selected_shots)
                    progress = 60

                    selected_YDM = request.form.get('detectionModel', 'YD1')
                    print(f'{selected_YDM} is  selecteeed:') #YD6
                    detected_objects, object_counts, total_frames = detect_yolo(output_path, selected_YDM)
                    progress = 70

                    selected_YSM = request.form.get('segmentationModel', 'YS1')
                    print(f'{selected_YSM} is  selecteeed:') #YD6
                    segmented_objects, segment_counts = segment_yolo(output_path, selected_YSM)
                    progress = 80

                    
                    # Check if the selected detection model is 'YD1' before proceeding with tracking
                    if selected_YDM == 'YD1':
                        selected_YTO = request.form.get('trackingModel', 'YT1')
                        print(f'{selected_YTO} is selected:')  # e.g., 'YT3'
                        tracked_objects, tracking_path = track_yolo(output_path, selected_YTO, timestamp)
                        progress = 90
                    else:
                        print("Tracking disabled as the selected detection model is not 'YD1'.")
                        # You can set tracked_objects and tracking_path to None or a default value
                        tracked_objects, tracking_path = None, None
                        progress = 90  # or adjust progress as needed

                    
                    ret = {}

                    ret['detected_objects'] = [y.dict() for y in detected_objects]
                    ret['segmented_objects'] = [y.dict() for y in segmented_objects]
                    # When creating the response, check if tracked_objects is not None before iterating
                    if tracked_objects is not None:
                        ret['tracking'] = [y.dict() for y in tracked_objects]
                    else:
                        ret['tracking'] = []  # or any other appropriate default value

                    json_output = json.dumps(ret, indent=4)

                    # Write to a json file
                    object_detected_file_path = f'/home/mehdi/03_GUI/static/generated_files/objects_{timestamp}.json'
                    with open(object_detected_file_path, 'w') as file:
                        file.write(json_output)

                    # Write the SmartCombo to a text file
                    output_txt_path = f'/home/mehdi/03_GUI/static/generated_files/output_file_{timestamp}.txt'
                    result = reduce_floats_and_convert_to_txt(object_detected_file_path, output_txt_path)

                    
                    # Initialize default values for all fields
                    metadata_date = ''
                    metadata_time = ''
                    metadata_team = ''
                    metadata_scorer = ''
                    metadata_shot_type = ''

                    # Check if a file is part of the request
                    if 'metadataFile' in request.files:
                        file = request.files['metadataFile']
                        if file.filename != '':
                            try:
                                # Read the file and parse JSON
                                file_content = json.load(file)
                                # Extract data from JSON
                                metadata_date = file_content.get('date', metadata_date)
                                metadata_time = file_content.get('time', metadata_time)
                                metadata_team = file_content.get('team', {}).get('value', metadata_team)
                                metadata_scorer = file_content.get('scorer', {}).get('value', metadata_scorer)
                                metadata_shot_type = file_content.get('shot type', {}).get('value', metadata_shot_type)
                            except json.JSONDecodeError:
                                return "Invalid JSON file."

                    # Overwrite with form data if present
                    metadata_date = request.form.get('metadataDate', metadata_date)
                    metadata_time = request.form.get('metadataTime', metadata_time)
                    metadata_team = request.form.get('metadataTeam', metadata_team)
                    metadata_scorer = request.form.get('metadataScorer', metadata_scorer)
                    metadata_shot_type = request.form.get('metadataShotType', metadata_shot_type)

                    # Construct the metadata object
                    metadata = {
                        "date": metadata_date,
                        "time": metadata_time,
                        "team": {"type": "team", "value": metadata_team},
                        "action": "goal",
                        "scorer": {"type": "player", "value": metadata_scorer},
                        "shot type": {"type": "goal shot type", "value": metadata_shot_type}
                    }

                    # Convert the dictionary to a JSON string
                    metadata_json = json.dumps(metadata, indent=4)

                    # Temporarily store the current contents of the file
                    with open(output_txt_path, 'r') as file:
                        current_contents = file.read()

                    # Write metadata and then the original contents back to the file
                    with open(output_txt_path, 'w') as file:
                        file.write("Metadata of the game:\n")
                        file.write(metadata_json + '\n\n')

                        if enable_transcript:
                            file.write("Automatic Speech Recognition of the clip: (Needed to be translated to ENGLISH)\n")
                            file.write(final_transcript + '\n\n')
                        else:
                            file.write("Automatic Speech Recognition of the clip: Disabled\n\n")

                        if enable_rme:
                            file.write("Root Mean Error of the audio of the clip. RMS is a measure of the average power of the audio signal: \n")
                            file.write(rms_values_str + '\n\n')
                        else:
                            file.write("Root Mean Error of the audio of the clip: Disabled\n\n")

                        file.write(current_contents)

                    # Run the filter function to remove frames with low detection from text file
                    filter_frames(output_txt_path, 6)

                    # Retrieve OpenAI API key from the form
                    openai_api_key = request.form.get('openai_api_key')
                    temperature = float(request.form.get('temperature', 0.7))
                    max_tokens = int(request.form.get('maxTokens', 1000))
                    top_p = int(request.form.get('topP', 1))
                    presence_penalty = int(request.form.get('presencePenalty', 0))
                    frequency_penalty = int(request.form.get('frequencyPenalty', 0))
                    seed_input = request.form.get('seed')
                    seed = int(seed_input) if seed_input.strip() else None  
                    print("Received API Key:", openai_api_key)  

                    # Path to your text file
                    text_file_path = output_txt_path
                    chat_response, chat_error = chat_with_openai(openai_api_key, text_file_path, temperature, 
                                                    max_tokens, top_p, presence_penalty, frequency_penalty,
                                                    seed)
                    progress = 100
                    # Check if there was an error in chat response
 
                    if chat_error:
                        print(f"Error during chat: {chat_error}")
                        chat_response = str(chat_error)  

                    # Incorporate these results into a response
                    response = {
                        "processed_video_path": processed_video_path,
                        "final_transcript": final_transcript,
                        "rms_values_str": rms_values_str,
                        "chat_response": chat_response,
                        "animation_file_path": animation_file_path,
                        "object_counts": object_counts,
                        "segment_counts": segment_counts,
                        "tracking_path": tracking_path,
                        "total_frames": total_frames
                    }
                    print(f"{segment_counts} object counts here")
                    return jsonify(response)
                    
                
    progress = 0 
    return render_template('index.html')


###########################################################################################
@app.route('/process-json', methods=['POST'])
def process_json():
    file_keys = ['metadataFile', 'metadataFile2']

    for key in file_keys:
        if key in request.files:
            file = request.files[key]
            if file.filename != '':
                try:
                    # Read the file and parse JSON
                    file_content = json.load(file)
                    return jsonify(file_content)
                except json.JSONDecodeError:
                    return jsonify({"error": "Invalid JSON file."}), 400

    return jsonify({"error": "No valid file uploaded."}), 400



@app.route('/progress')
def get_progress():
    return jsonify({"progress": progress})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
############################################################################################


@app.route('/handle-selected-frames', methods=['GET', 'POST'])
def handle_selected_frames():
    global progress  
    # Current timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    
    data = request.json
    selected_frames = data.get('selectedFrames', [])
    print(f"selected frames is here {selected_frames}")
    
    # Process selected frames and create a video
    video_path_selective_frames = process_selected_frames(selected_frames, timestamp)

    print(f"The path to the created video is here:{video_path_selective_frames} ")
    ####################################################################
    #getting the values from the UI for second template
    data = request.json
    user_inputs = data.get('userInput', {})


    enable_transcript = data.get('enableTranscript', False)
    enable_RME = data.get('enableRMEAudio', False)

    processed_video_path = data.get('processedVideo', '')
    
    if enable_transcript:
        # Process video transcript
        model_name = user_inputs.get('whisper_model', '')
        print(f'{model_name} is  selecteeed:') #YD6
        final_transcript, audio_path = process_video_transcript(processed_video_path, timestamp, model_name)
        print(f"The Transcript = {final_transcript} \n From the {processed_video_path}")
        progress = 40
    if enable_RME:
        # Analyze the RME of the Audio
        frame_length= user_inputs.get('frame_length', '')
        print(f'{frame_length} is  selecteeed:') #YD6

        hop_length= user_inputs.get('hop_length', '')
        print(f'{hop_length} is  selecteeed:') #YD6
        rms_values_str, animation_file_path = RME_audio(audio_path, timestamp, frame_length, hop_length)
        print(f'The RME value of the the Audio of the video clip is = \n {rms_values_str}')
        progress = 50
    if video_path_selective_frames:
        #object detection
        selected_YDM = user_inputs.get('selected_YDM', '')
        print(f'{selected_YDM} is  selecteeed:') #YD6
        detected_objects, object_counts, total_frames = detect_yolo(video_path_selective_frames, selected_YDM)
        progress = 60

        #segmentation
        selected_YSM = user_inputs.get('selected_YSM', '')
        print(f'{selected_YSM} is  selecteeed:') #YD6
        segmented_objects, segment_counts = segment_yolo(video_path_selective_frames, selected_YSM)
        progress = 70

        if selected_YDM =='YD1':
            selected_YTO = user_inputs.get('selected_YTO', '')
            print(f'{selected_YTO} is  selecteeed:') #YD6
            tracked_objects, tracking_path = track_yolo(video_path_selective_frames, selected_YTO, timestamp)
            progress = 80
        else:
            print("Tracking disabled as the selected detection model is not 'YD1'.")
            tracked_objects, tracking_path = None, None
            progress = 80  # or adjust progress as needed

        
        ret = {}

        ret['detected_objects'] = [y.dict() for y in detected_objects]
        ret['segmented_objects'] = [y.dict() for y in segmented_objects]
        if tracked_objects is not None:
            ret['tracking'] = [y.dict() for y in tracked_objects]
        else:
            ret['tracking'] = []  

        json_output = json.dumps(ret, indent=4)

        # Write to a json file
        object_detected_file_path = f'/home/mehdi/03_GUI/static/generated_files/objects_{timestamp}.json'
        with open(object_detected_file_path, 'w') as file:
            file.write(json_output)

        # Write the SmartCombo to a text file
        output_txt_path = f'/home/mehdi/03_GUI/static/generated_files/output_file_{timestamp}.txt'
        result = reduce_floats_and_convert_to_txt(object_detected_file_path, output_txt_path)


        
        # Initialize default values for all fields
        metadata_date = ''
        metadata_time = ''
        metadata_team = ''
        metadata_scorer = ''
        metadata_shot_type = ''

        # Check if a file is part of the request
        if 'metadataFile2' in request.files:
            file = request.files['metadataFile2']
            if file.filename != '':
                try:
                    # Read the file and parse JSON
                    file_content = json.load(file)
                    # Extract data from JSON
                    metadata_date = file_content.get('date', metadata_date)
                    metadata_time = file_content.get('time', metadata_time)
                    metadata_team = file_content.get('team', {}).get('value', metadata_team)
                    metadata_scorer = file_content.get('scorer', {}).get('value', metadata_scorer)
                    metadata_shot_type = file_content.get('shot type', {}).get('value', metadata_shot_type)
                except json.JSONDecodeError:
                    return "Invalid JSON file."

        # Overwrite with form data if present
        metadata_date = user_inputs.get(metadata_date, '')
        metadata_time = user_inputs.get(metadata_time, '')
        metadata_team = user_inputs.get(metadata_team, '')
        metadata_scorer = user_inputs.get(metadata_scorer, '')
        metadata_shot_type = user_inputs.get(metadata_shot_type, '')

        # Construct the metadata object
        metadata = {
            "date": metadata_date,
            "time": metadata_time,
            "team": {"type": "team", "value": metadata_team},
            "action": "goal",
            "scorer": {"type": "player", "value": metadata_scorer},
            "shot type": {"type": "goal shot type", "value": metadata_shot_type}
        }

        # Convert the dictionary to a JSON string
        metadata_json = json.dumps(metadata, indent=4)

        # Temporarily store the current contents of the file
        with open(output_txt_path, 'r') as file:
            current_contents = file.read()

        # Write metadata and then the original contents back to the file
        with open(output_txt_path, 'w') as file:
            file.write("Metadata of the game:\n")
            file.write(metadata_json + '\n\n')

            if enable_transcript:
                file.write("Automatic Speech Recognition of the clip: (Needed to be translated to ENGLISH)\n")
                file.write(final_transcript + '\n\n')
            else:
                file.write("Automatic Speech Recognition of the clip: Disabled\n\n")

            if enable_RME:
                file.write("Root Mean Error of the audio of the clip. RMS is a measure of the average power of the audio signal: \n")
                file.write(rms_values_str + '\n\n')
            else:
                file.write("Root Mean Error of the audio of the clip: Disabled\n\n")

            file.write(current_contents)

        progress = 90
        # Retrieve OpenAI API key from the form
        openai_api_key = user_inputs.get('openai_api_key', '')
        print(f"for 1000 time api key is here {openai_api_key}")

        temperature = float(user_inputs.get('temperature', 0.7))

        max_tokens = int(user_inputs.get('max_tokens', 1000))

        top_p = int(user_inputs.get('top_p', 1))

        presence_penalty = int(user_inputs.get('presence_penalty', 0))

        frequency_penalty = int(user_inputs.get('frequency_penalty', 0))
        
        seed_input = user_inputs.get('seed_input', '')

        seed = int(seed_input) if seed_input.isdigit() else None 

        # Path to your text file
        text_file_path = output_txt_path
        chat_response, chat_error = chat_with_openai(openai_api_key, text_file_path, temperature, 
                                        max_tokens, top_p, presence_penalty, frequency_penalty,
                                        seed)
        # Check if there was an error in chat response

        if chat_error:
            print(f"Error during chat: {chat_error}")
            chat_response = str(chat_error)  
        progress = 100

        # Incorporate these results into a response
        response = {
            "processed_video_path": processed_video_path,
            "final_transcript": final_transcript,
            "rms_values_str": rms_values_str,
            "chat_response": chat_response,
            "animation_file_path": animation_file_path,
            "object_counts": object_counts,
            "segment_counts": segment_counts,
            "tracking_path": tracking_path,
            "total_frames": total_frames
        }
        return jsonify(response)
            
        
    progress = 0 
    return render_template('index.html')



    
def process_selected_frames(frame_paths, timestamp):
    frame_paths.sort(key=lambda x: int(re.search(r"_(\d+)\.jpg$", x).group(1)))

    frame_list_file = f'/home/mehdi/03_GUI/static/selected_frames_list_{timestamp}.txt'
    with open(frame_list_file, 'w') as file:
        for path in frame_paths:
            # Remove the 'static/' prefix
            modified_path = path.replace('static/', '')
            file.write(f"file '{modified_path}'\n")
            file.write("duration 1\n")  # Specify duration for each frame


    video_path_selective_frames = create_video_from_frames(frame_list_file, timestamp)

    return video_path_selective_frames




def create_video_from_frames(frame_list_file, timestamp):
    video_path_selective_frames = f'/home/mehdi/03_GUI/static/video_from_selection/selective_video_{timestamp}.mp4'

    # Modify the paths to be relative from the static directory
    frame_list_file_path = f'{frame_list_file}'
    ffmpeg_command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', frame_list_file_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-r', '1',  # Frame rate
        video_path_selective_frames
    ]

    try:
        subprocess.run(ffmpeg_command, check=True, cwd='/home/mehdi/03_GUI/static/')
        print("Video created at:", video_path_selective_frames)
    except subprocess.CalledProcessError as e:
        print("Error during video creation:", e)
        return "Error during video creation"

    return video_path_selective_frames








################################################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000 ,debug=True)
