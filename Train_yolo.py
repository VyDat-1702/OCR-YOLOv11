import os
from ultralytics import YOLO
import yaml
import argparse

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    defaults = {
        'model_name': 'yolo11n.pt',
        'data': 'yolo_data/data.yaml',
        'epochs': 120,
        'imgsz': 320,
        'batch': 4,
        'cache': True,
        'patience': 20,
        'plots': True,
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_yolo_config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    model_name = config.pop('model_name')
    model = YOLO(model_name)
    
    results = model.train(**config)

if __name__ == '__main__':
    main()