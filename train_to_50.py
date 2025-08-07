#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
50エポックまで学習するスクリプト
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def train_to_50():
    """50エポックまで学習"""
    print("50エポックまで学習します（現在39エポック完了）...")
    
    # 最後のモデルから継続
    model = YOLO('/Users/iinuma/Desktop/自動ROI/yolo_dataset/models/lesion_detection_v2/weights/last.pt')
    
    # 残り11エポック学習
    results = model.train(
        data='/Users/iinuma/Desktop/自動ROI/yolo_dataset/dataset.yaml',
        epochs=11,  # 39 + 11 = 50
        batch=4,
        imgsz=640,
        patience=20,
        device='cpu',
        project='/Users/iinuma/Desktop/自動ROI/yolo_dataset/models',
        name='lesion_detection_v2_50epochs',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    print("✅ 50エポック完了")
    return results

if __name__ == '__main__':
    train_to_50()