#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO学習用のアノテーション作成を支援するスクリプト
既存のマスク済み画像と手動ROIツールの結果を活用
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import random
import argparse

class AnnotationHelper:
    """アノテーション作成支援クラス"""
    
    def __init__(self):
        self.class_mapping = {
            'ADM': 0,
            'Ephelis': 1,
            'Melasma': 2,
            'Solar_lentigo': 3,
            'Solar lentigo': 3,  # エイリアス
            'Nevus': 4,
            'Basal_cell_carcinoma': 5,
            'Basal cell carcinoma': 5,  # エイリアス
            'Seborrheic_keratosis': 6,
            'Seborrheic keratosis': 6,  # エイリアス
            'Malignant_melanoma': 7,
            'Malignant melanoma': 7,  # エイリアス
        }
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        バウンディングボックスをYOLO形式に変換
        [x1, y1, x2, y2] -> [x_center, y_center, width, height] (正規化)
        """
        x1, y1, x2, y2 = bbox
        
        # 中心座標を計算
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        
        # 幅と高さを計算
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return x_center, y_center, width, height
    
    def create_manual_annotations(self, roi_results_path, images_dir, output_dir):
        """
        手動ROI切り取りツールの結果からアノテーションを作成
        
        Args:
            roi_results_path: ROI結果JSONファイルのパス
            images_dir: 元画像のディレクトリ
            output_dir: 出力ディレクトリ
        """
        # ROI結果を読み込み
        with open(roi_results_path, 'r', encoding='utf-8') as f:
            roi_data = json.load(f)
        
        output_path = Path(output_dir)
        
        annotations = []
        
        for item in tqdm(roi_data, desc="アノテーション作成中"):
            image_path = Path(item['image'])
            if not image_path.exists():
                # 相対パスの場合の処理
                image_path = Path(images_dir) / image_path.name
            
            if not image_path.exists():
                print(f"画像が見つかりません: {image_path}")
                continue
            
            # 画像を読み込んで寸法を取得
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # カテゴリを取得
            category = image_path.parent.name
            if category not in self.class_mapping:
                print(f"不明なカテゴリ: {category}")
                continue
            
            class_id = self.class_mapping[category]
            
            # YOLOアノテーションを作成
            yolo_annotations = []
            for roi in item['rois']:
                bbox = roi['bbox']
                x_center, y_center, width, height = self.convert_bbox_to_yolo(bbox, w, h)
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # アノテーション情報を保存
            annotations.append({
                'image_path': str(image_path),
                'category': category,
                'class_id': class_id,
                'annotations': yolo_annotations,
                'num_lesions': len(yolo_annotations)
            })
        
        return annotations
    
    def split_dataset(self, annotations, train_ratio=0.7, valid_ratio=0.2):
        """
        データセットを学習用、検証用、テスト用に分割
        
        Args:
            annotations: アノテーションリスト
            train_ratio: 学習データの割合
            valid_ratio: 検証データの割合
        """
        # シャッフル
        random.shuffle(annotations)
        
        total = len(annotations)
        train_size = int(total * train_ratio)
        valid_size = int(total * valid_ratio)
        
        train_data = annotations[:train_size]
        valid_data = annotations[train_size:train_size + valid_size]
        test_data = annotations[train_size + valid_size:]
        
        return train_data, valid_data, test_data
    
    def save_yolo_dataset(self, annotations, dataset_dir, split='train'):
        """
        YOLO形式でデータセットを保存
        
        Args:
            annotations: アノテーションリスト
            dataset_dir: データセットのルートディレクトリ
            split: 'train', 'valid', 'test'のいずれか
        """
        dataset_path = Path(dataset_dir)
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for ann in tqdm(annotations, desc=f"{split}データ作成中"):
            # 画像をコピー
            src_image = Path(ann['image_path'])
            dst_image = images_dir / src_image.name
            shutil.copy2(src_image, dst_image)
            
            # ラベルファイルを作成
            label_file = labels_dir / (src_image.stem + '.txt')
            with open(label_file, 'w') as f:
                for line in ann['annotations']:
                    f.write(line + '\n')
        
        print(f"✅ {split}データ作成完了: {len(annotations)}枚")
    
    def create_dataset_from_manual_roi(self, roi_results_path, images_dir, dataset_dir):
        """
        手動ROI結果から完全なYOLOデータセットを作成
        """
        print("=== YOLOデータセット作成 ===\n")
        
        # アノテーションを作成
        annotations = self.create_manual_annotations(roi_results_path, images_dir, dataset_dir)
        print(f"総アノテーション数: {len(annotations)}")
        
        # カテゴリごとの統計を表示
        category_counts = {}
        for ann in annotations:
            cat = ann['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nカテゴリごとの画像数:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")
        
        # データセットを分割
        train_data, valid_data, test_data = self.split_dataset(annotations)
        print(f"\nデータセット分割:")
        print(f"  学習用: {len(train_data)}枚")
        print(f"  検証用: {len(valid_data)}枚")
        print(f"  テスト用: {len(test_data)}枚")
        
        # 各分割を保存
        self.save_yolo_dataset(train_data, dataset_dir, 'train')
        self.save_yolo_dataset(valid_data, dataset_dir, 'valid')
        self.save_yolo_dataset(test_data, dataset_dir, 'test')
        
        print(f"\n✅ データセット作成完了: {dataset_dir}")

def main():
    parser = argparse.ArgumentParser(description='YOLO学習用データセット作成支援')
    parser.add_argument('--roi-results', type=str, required=True,
                        help='手動ROI結果のJSONファイル')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='元画像のディレクトリ')
    parser.add_argument('--output-dir', type=str, default='yolo_dataset',
                        help='出力データセットディレクトリ')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='学習データの割合')
    parser.add_argument('--valid-ratio', type=float, default=0.2,
                        help='検証データの割合')
    
    args = parser.parse_args()
    
    # アノテーション支援ツールを初期化
    helper = AnnotationHelper()
    
    # データセットを作成
    helper.create_dataset_from_manual_roi(
        args.roi_results,
        args.images_dir,
        args.output_dir
    )

if __name__ == "__main__":
    main()