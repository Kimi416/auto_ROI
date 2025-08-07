#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
シンプルなYOLOv8学習スクリプト
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def create_config():
    """YOLO設定ファイル作成"""
    config = {
        'path': '/Users/iinuma/Desktop/自動ROI/yolo_dataset',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 8,
        'names': {
            0: 'Melasma',
            1: 'Solar_lentigo', 
            2: 'Nevus',
            3: 'ADM',
            4: 'Ephelis',
            5: 'Seborrheic_keratosis',
            6: 'Basal_cell_carcinoma',
            7: 'Malignant_melanoma'
        }
    }
    
    config_path = Path('/Users/iinuma/Desktop/自動ROI/yolo_dataset/dataset.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return str(config_path)

def train_model():
    """YOLOv8モデル学習"""
    print("YOLOv8学習開始...")
    
    # 設定ファイル作成
    config_path = create_config()
    print(f"設定ファイル: {config_path}")
    
    # モデル初期化
    model = YOLO('yolov8n.pt')
    
    # 学習実行
    results = model.train(
        data=config_path,
        epochs=100,
        batch=4,
        imgsz=640,
        patience=20,
        device='cpu',
        project='/Users/iinuma/Desktop/自動ROI/yolo_dataset/models',
        name='lesion_detection_v2',
        exist_ok=True,
        verbose=True,
        save_period=10,
        plots=True
    )
    
    print("✅ 学習完了")
    return results

if __name__ == '__main__':
    train_model()