import json
import os
import shutil
from sklearn.model_selection import train_test_split

images_dir = '../train_dataset/images'
json_path = '../train_dataset/annotations/instances_default.json'
output_dir = '../split_dataset/'
train_rate = 0.7

with open(json_path, 'r') as file:
    coco = json.load(file)

os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/valid'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

image_files = [img['file_name'] for img in coco['images']]
train_files, valid_files = train_test_split(image_files, test_size=(1 - train_rate))

for file in train_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(output_dir, 'images/train', file))
for file in valid_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(output_dir, 'images/valid', file))


def filter_annotations(files, coco):
    image_ids = [img['id'] for img in coco['images'] if img['file_name'] in files]
    images = [img for img in coco['images'] if img['file_name'] in files]
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] in image_ids]
    return {
        'images': images,
        'annotations': annotations,
        'categories': coco['categories']
    }


train_coco = filter_annotations(train_files, coco)
val_coco = filter_annotations(valid_files, coco)

with open(os.path.join(output_dir, 'annotations/instances_train.json'), 'w') as file:
    json.dump(train_coco, file)

with open(os.path.join(output_dir, 'annotations/instances_valid.json'), 'w') as file:
    json.dump(val_coco, file)


def convert_to_yolo_format(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def save_yolo_labels(coco_data, output_dir):
    for image_data in coco_data['images']:
        image_id = image_data['id']
        img_width = image_data['width']
        img_height = image_data['height']
        file_name = image_data['file_name']

        yolo_labels = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                bbox = annotation['bbox']
                category_id = annotation['category_id']

                x_center, y_center, width, height = convert_to_yolo_format(bbox, img_width, img_height)

                yolo_labels.append(f"{category_id} {x_center} {y_center} {width} {height}\n")

        labels_dir = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
        with open(labels_dir, 'w') as label_file:
            label_file.writelines(yolo_labels)


os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/valid'), exist_ok=True)

save_yolo_labels(train_coco, f"{output_dir}labels/train/")
save_yolo_labels(val_coco, f"{output_dir}labels/valid/")
