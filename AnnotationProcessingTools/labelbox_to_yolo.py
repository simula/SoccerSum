
import json
import os

def convert_to_yolo_bbox(bbox, img_width, img_height):
    """ Convert bounding box to YOLO format. """
    x_center = bbox['left'] + (bbox['width'] / 2)
    y_center = bbox['top'] + (bbox['height'] / 2)
    width = bbox['width']
    height = bbox['height']

    # Normalize by image size
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return f"{x_center} {y_center} {width} {height}"

def convert_to_yolo_polygon(polygon, img_width, img_height):
    """ Convert polygon to YOLO format. """
    points = []
    for point in polygon:
        x = point['x'] / img_width
        y = point['y'] / img_height
        points.extend([x, y])

    return ' '.join(map(str, points))

# Class name to ID mapping
detection_classes = {
    "Player": 0,
    "Goalkeeper": 1,
    "Referee": 2,
    "Ball": 3,
    "Logo": 4,
    "Penalty Mark": 5,
    "Corner Flagpost": 6,
    "Goal Net": 7
}

segmentation_classes = {
    "PenaltyBox": 0,
    "GoalBox": 1
}

def process_ndjson(file_path):
    os.makedirs('detection', exist_ok=True)
    os.makedirs('segmentation', exist_ok=True)

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            img_width = data['media_attributes']['width']
            img_height = data['media_attributes']['height']
            external_id = data['data_row']['external_id'].split('.')[0]

            bbox_annotations = []
            polygon_annotations = []

            for label in data['projects'].values():
                for annotation in label['labels'][0]['annotations']['objects']:
                    class_name = annotation['name']
                    if annotation['annotation_kind'] == 'ImageBoundingBox':
                        class_id = detection_classes.get(class_name, -1)
                        if class_id != -1:
                            yolo_bbox = convert_to_yolo_bbox(annotation['bounding_box'], img_width, img_height)
                            bbox_annotations.append(f"{class_id} {yolo_bbox}")

                    elif annotation['annotation_kind'] == 'ImagePolygon':
                        class_id = segmentation_classes.get(class_name, -1)
                        if class_id != -1:
                            yolo_polygon = convert_to_yolo_polygon(annotation['polygon'], img_width, img_height)
                            polygon_annotations.append(f"{class_id} {yolo_polygon}")

            with open(f'detection/{external_id}.txt', 'w') as bbox_file:
                bbox_file.write('\n'.join(bbox_annotations))
            
            with open(f'segmentation/{external_id}.txt', 'w') as polygon_file:
                polygon_file.write('\n'.join(polygon_annotations))

# Usage
process_ndjson('/home/mehdi/YOLO_Train/Create_Dataset/January19- 2023/Export v2 project - 2023 - Ellitesereian 250 - 1_20_2024.ndjson')
