import os
import xml.etree.ElementTree as ET
import yaml
from sklearn.model_selection import train_test_split
import shutil
import argparse

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    defaults = {
        'data_path': 'SceneTrialTrain',
        'xml_path': os.path.join('SceneTrialTrain', 'word.xml'),
        'save_yolo_data_dir': 'yolo_data',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'seed': 42,
        'shuffle': True,
        'class_id': 0,
        'class_name': 'text'
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config
    
def get_xml_data(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    data_xml = ET.parse(xml_path)
    root = data_xml.getroot()
    image_paths = []        
    image_sizes = []
    bounding_boxes = []
    image_labels = []
    
    for image in root:
        box = []
        label = []
        
        w, h = image[1].attrib['x'], image[1].attrib['y']
        
        if image[2].text and ("Ã©" in image[2].text.lower() or "Ã±" in image[2].text.lower()):
            continue
            
        for bbox in image[2]:
            x = bbox.attrib['x']
            y = bbox.attrib['y']
            wb = bbox.attrib['width']
            hb = bbox.attrib['height']

            box.append([
                float(x),
                float(y),
                float(wb),
                float(hb)
            ])
            label.append(bbox[0].text)
        
        image_paths.append(image[0].text)
        image_sizes.append((float(w), float(h)))
        bounding_boxes.append(box)
        image_labels.append(label)
        
    return image_paths, image_sizes, bounding_boxes, image_labels

def convert_to_yolo(image_paths, image_sizes, bounding_boxes, class_id=0):
    yolo_data = []
    
    for img_path, img_sz, bb in zip(image_paths, image_sizes, bounding_boxes):
        labels = []
        width, height = img_sz
        
        for box in bb:
            x, y, w, h = box
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            
            label = f'{class_id} {x_center} {y_center} {w_norm} {h_norm}'
            labels.append(label)
            
        yolo_data.append((img_path, labels))
        
    return yolo_data

def split_data(yolo_data, seed=42, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True):
    train_data, temp_data = train_test_split(
        yolo_data,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=shuffle,
    )

    relative_test_size = test_size / (val_size + test_size)
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=relative_test_size,
        random_state=seed,
        shuffle=shuffle,
    )
    
    return train_data, val_data, test_data

def save_data(data, src_dir, target_dir):
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)
    
    for img_path, labels in data:
        src_img_path = os.path.join(src_dir, img_path)
        
        if not os.path.exists(src_img_path):
            continue
            
        shutil.copy(src_img_path, os.path.join(target_dir, 'images'))
        
        image_name = os.path.basename(img_path)
        image_name = os.path.splitext(image_name)[0]
        
        with open(os.path.join(target_dir, "labels", f'{image_name}.txt'), 'w') as f:
            for lb in labels:
                f.write(f'{lb}\n')

def create_yaml(path, class_name='text'):
    data_yaml = {
        'path': os.path.abspath(path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images', 
        'nc': 1,
        'names': [class_name],
    }

    yolo_yaml_path = os.path.join(path, 'data.yaml')
    with open(yolo_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='create_yolo_data_config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    image_paths, image_sizes, bounding_boxes, image_labels = get_xml_data(config['xml_path'])
    yolo_data = convert_to_yolo(image_paths, image_sizes, bounding_boxes, config['class_id'])
    
    train_data, val_data, test_data = split_data(
        yolo_data,
        seed=config['seed'],
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size'],
        shuffle=config['shuffle']
    )
    
    os.makedirs(config['save_yolo_data_dir'], exist_ok=True)
    save_train_dir = os.path.join(config['save_yolo_data_dir'], "train")
    save_val_dir = os.path.join(config['save_yolo_data_dir'], "val")
    save_test_dir = os.path.join(config['save_yolo_data_dir'], "test")

    save_data(train_data, config['data_path'], save_train_dir)
    save_data(val_data, config['data_path'], save_val_dir)
    save_data(test_data, config['data_path'], save_test_dir)
    
    create_yaml(config['save_yolo_data_dir'], config['class_name'])

if __name__ == "__main__":
    main()