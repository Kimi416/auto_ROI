#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAD-UFES-20データセットを使用した皮膚病変検出モデルの訓練スクリプト
PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime
import json
import cv2
import numpy as np
from tqdm import tqdm

class PADUFESTrainer:
    """PAD-UFES-20データセットでのYOLO訓練クラス"""
    
    def __init__(self, dataset_path="pad_ufes_20", output_dir="runs/detect"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.classes = [
            'ACK',  # Actinic keratosis
            'BCC',  # Basal cell carcinoma
            'MEL',  # Melanoma
            'NEV',  # Nevus
            'SCC',  # Squamous cell carcinoma
            'SEK'   # Seborrheic keratosis
        ]
        
    def setup_dataset_structure(self):
        """PAD-UFES-20データセットのYOLO形式への変換"""
        print("📁 PAD-UFES-20データセット構造を設定中...")
        
        # YOLOフォーマット用ディレクトリ作成
        yolo_dataset = self.dataset_path / "yolo_format"
        for split in ['train', 'val', 'test']:
            (yolo_dataset / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dataset / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        return yolo_dataset
    
    def create_dataset_yaml(self, yolo_dataset_path):
        """YOLO用のデータセット設定ファイルを作成"""
        dataset_config = {
            'path': str(yolo_dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = yolo_dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ データセット設定ファイル作成: {yaml_path}")
        return yaml_path
    
    def convert_annotations_to_yolo(self, image_path, annotations, img_width, img_height):
        """アノテーションをYOLO形式に変換"""
        yolo_annotations = []
        
        for ann in annotations:
            # PAD-UFES-20のアノテーション形式に応じて調整
            class_id = self.classes.index(ann['class']) if ann['class'] in self.classes else 0
            
            # バウンディングボックスの正規化 (x_center, y_center, width, height)
            x_center = (ann['x'] + ann['width'] / 2) / img_width
            y_center = (ann['y'] + ann['height'] / 2) / img_height
            width = ann['width'] / img_width
            height = ann['height'] / img_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def prepare_training_data(self):
        """PAD-UFES-20データをYOLO訓練用に準備"""
        print("🔄 PAD-UFES-20データをYOLO形式に変換中...")
        
        yolo_dataset = self.setup_dataset_structure()
        
        # メタデータファイルの読み込み（PAD-UFES-20データセットの構造に応じて調整）
        metadata_path = self.dataset_path / "metadata.csv"
        if not metadata_path.exists():
            print(f"⚠️  メタデータファイルが見つかりません: {metadata_path}")
            print("PAD-UFES-20データセットを正しいディレクトリに配置してください")
            return None
        
        # データセット分割比率
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        # 簡単な例: 画像とアノテーションをコピー
        # 実際のPAD-UFES-20データセットの構造に応じて調整が必要
        images_dir = self.dataset_path / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            # データセット分割
            total_images = len(image_files)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            
            for i, img_path in enumerate(tqdm(image_files, desc="データ変換中")):
                # 分割決定
                if i < train_count:
                    split = 'train'
                elif i < train_count + val_count:
                    split = 'val'
                else:
                    split = 'test'
                
                # 画像コピー
                dst_img = yolo_dataset / split / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # ダミーラベル作成（実際のアノテーションデータがある場合は置き換え）
                label_path = yolo_dataset / split / 'labels' / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    # ダミーラベル: 画像中央に小さな病変があると仮定
                    f.write("0 0.5 0.5 0.1 0.1\n")
        
        yaml_path = self.create_dataset_yaml(yolo_dataset)
        return yaml_path
    
    def train_model(self, yaml_path, epochs=100, img_size=640, batch_size=16):
        """YOLOモデルの訓練"""
        print(f"🚀 PAD-UFES-20データセットでの訓練開始...")
        print(f"エポック数: {epochs}, 画像サイズ: {img_size}, バッチサイズ: {batch_size}")
        
        # YOLOv8nモデルを使用（軽量で高速）
        model = YOLO('yolov8n.pt')
        
        # 訓練設定
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=f'pad_ufes_20_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=10,
            save=True,
            cache=True,
            device='mps',  # macOS Metal Performance Shaders
            workers=4,
            project=self.output_dir
        )
        
        print("✅ 訓練完了!")
        return results
    
    def validate_model(self, model_path, yaml_path):
        """訓練されたモデルの検証"""
        print("📊 モデル検証中...")
        
        model = YOLO(model_path)
        results = model.val(data=yaml_path)
        
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def extract_lesions_from_image(self, model_path, image_path, output_dir, confidence_threshold=0.25):
        """訓練されたモデルを使用して画像から病変を抽出"""
        print(f"🔍 病変抽出開始: {image_path}")
        
        model = YOLO(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 推論実行
        results = model(image_path, conf=confidence_threshold)
        
        # 元画像読み込み
        image = cv2.imread(str(image_path))
        image_name = Path(image_path).stem
        
        extracted_lesions = []
        
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                for j, box in enumerate(boxes):
                    # バウンディングボックス座標
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes[class_id]
                    
                    # 病変領域を抽出
                    lesion_roi = image[y1:y2, x1:x2]
                    
                    # 抽出した病変を保存
                    lesion_filename = f"{image_name}_lesion_{j+1}_{class_name}_{confidence:.3f}.jpg"
                    lesion_path = output_dir / lesion_filename
                    cv2.imwrite(str(lesion_path), lesion_roi)
                    
                    extracted_lesions.append({
                        'filename': lesion_filename,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'area': (x2-x1) * (y2-y1)
                    })
                    
                    print(f"  抽出 {j+1}: {class_name} (信頼度: {confidence:.3f})")
        
        # 抽出結果の保存
        result_json = output_dir / f"{image_name}_extraction_results.json"
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(extracted_lesions, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 病変抽出完了: {len(extracted_lesions)}個の病変を抽出")
        return extracted_lesions

def main():
    """メイン実行関数"""
    print("🎯 PAD-UFES-20 皮膚病変検出モデル訓練システム")
    print("=" * 60)
    
    trainer = PADUFESTrainer()
    
    # 1. データセット準備
    yaml_path = trainer.prepare_training_data()
    if yaml_path is None:
        print("❌ データセット準備に失敗しました")
        return
    
    # 2. モデル訓練
    training_results = trainer.train_model(yaml_path, epochs=50)
    
    # 3. 最良モデルのパスを取得
    best_model_path = training_results.save_dir / 'weights' / 'best.pt'
    
    # 4. モデル検証
    validation_results = trainer.validate_model(best_model_path, yaml_path)
    
    # 5. 病変抽出テスト
    test_image = "test.jpg"  # テスト画像のパス
    if Path(test_image).exists():
        extracted_lesions = trainer.extract_lesions_from_image(
            best_model_path, 
            test_image, 
            "extracted_lesions_pad_ufes",
            confidence_threshold=0.3
        )
        
        print(f"\n🎉 病変抽出テスト完了: {len(extracted_lesions)}個の病変を検出")
    else:
        print(f"⚠️  テスト画像が見つかりません: {test_image}")
    
    print("\n✅ PAD-UFES-20訓練・抽出プロセス完了!")

if __name__ == "__main__":
    main()