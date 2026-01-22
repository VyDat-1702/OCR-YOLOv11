import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import yaml
import argparse

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    defaults = {
        'data_path': 'SceneTrialTrain',
        'xml_path': os.path.join('SceneTrialTrain', 'word.xml'),
        'save_dir': 'ocr_dataset',
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def get_xml_data(data_path, xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    data_xml = ET.parse(xml_path)
    root = data_xml.getroot()
    img_paths = []
    img_sizes = []
    bounding_boxes = []
    image_labels = []
    
    for image in root: 
        img_name = image.find('imageName')
        size = image.find('resolution')
        width = size.attrib['x']
        height = size.attrib['y']
        
        bb = []
        label = []
        
        for bboxes in image.findall('taggedRectangles'):
            for box in bboxes:
                x = float(box.attrib['x'])
                y = float(box.attrib['y'])
                bw = float(box.attrib['width'])
                bh = float(box.attrib['height'])
                
                bb.append([x, y, bw, bh])
                label.append(box.find('tag').text)
        
        img_paths.append(os.path.join(data_path, img_name.text))
        img_sizes.append((width, height))
        bounding_boxes.append(bb)
        image_labels.append(label)

    return img_paths, img_sizes, bounding_boxes, image_labels

def split_bounding_box(image_paths, image_labels, bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    labels = []
    
    for path, img_label, bbs in zip(image_paths, image_labels, bboxes):
        if not os.path.exists(path):
            print(f"Warning: Image not found: {path}")
            continue
            
        image = Image.open(path)
        
        for lb, bbox in zip(img_label, bbs):
            x, y, w, h = bbox
            cropped_image = image.crop((x, y, x + w, y + h))
            
            cropped_array = np.array(cropped_image)
            mean_intensity = np.mean(cropped_array)
            
            if mean_intensity < 35 or mean_intensity > 220:
                continue
            if cropped_image.size[0] < 10 or cropped_image.size[1] < 10:
                continue
            
            filename = f'{count:06d}.jpg'
            new_img_path = os.path.join(save_dir, filename)
            cropped_image.save(new_img_path)
            label = new_img_path + '\t' + lb
            labels.append(label)
            count += 1
    
    print(f'Created {count} images')
    
    with open(os.path.join(save_dir, 'labels.txt'), 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f'{label}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='create_ocr_data.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    img_paths, img_sizes, bboxes, image_labels = get_xml_data(
        config['data_path'], 
        config['xml_path']
    )
    split_bounding_box(img_paths, image_labels, bboxes, config['save_dir'])

if __name__ == '__main__':
    main()