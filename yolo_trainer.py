#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8病変検出システム
- アノテーションデータからYOLOデータセット作成
- YOLOv8学習実行
- 検出精度テスト
"""

import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import random
import argparse
import json
from datetime import datetime

class LesionYOLOTrainer:
    def __init__(self, images_dir, annotations_dir, output_dir="yolo_dataset"):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        
        # 病変タイプ定義
        self.disease_classes = [
            'Melasma',              # 0. 肝斑
            'Solar_lentigo',        # 1. 日光性色素斑
            'Nevus',                # 2. 母斑
            'ADM',                  # 3. 後天性真皮メラノサイトーシス
            'Ephelis',              # 4. 雀卵斑
            'Seborrheic_keratosis', # 5. 脂漏性角化症
            'Basal_cell_carcinoma', # 6. 基底細胞癌（BCC）
            'Malignant_melanoma'    # 7. 悪性黒色腫
        ]
        
        self.setup_directories()
    
    def setup_directories(self):
        """YOLOデータセット構造を作成"""
        print("YOLOデータセット構造を作成中...")
        
        # メインディレクトリ
        self.output_dir.mkdir(exist_ok=True)
        
        # 分割データセット
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 結果保存用
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        
        print(f"📁 出力先: {self.output_dir}")
    
    def create_dataset_config(self):
        """YOLO設定ファイル作成"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(self.disease_classes),
            'names': {i: name for i, name in enumerate(self.disease_classes)}
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📝 設定ファイル作成: {config_path}")
        return config_path
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """データセット分割とコピー"""
        print("データセット準備中...")
        
        # アノテーション済み画像とラベルを取得
        label_files = list((self.annotations_dir / 'labels').glob('*.txt'))
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        valid_pairs = []
        for label_file in label_files:
            # 対応する画像ファイルを探す（再帰的に検索）
            base_name = label_file.stem
            image_file = None
            
            # organized_masked内の全サブディレクトリから画像を探す
            for ext in image_extensions:
                # パターン: base_name.ext
                potential_images = list(self.images_dir.rglob(f"{base_name}{ext}"))
                if potential_images:
                    image_file = potential_images[0]  # 最初に見つかった画像
                    break
            
            if image_file:
                valid_pairs.append((image_file, label_file))
        
        print(f"有効な画像-ラベルペア: {len(valid_pairs)}組")
        
        if len(valid_pairs) == 0:
            raise ValueError("有効な画像-ラベルペアが見つかりません")
        
        # データ分割（シンプルな実装）
        random.seed(42)
        shuffled_pairs = valid_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        total = len(shuffled_pairs)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)
        train_size = total - test_size - val_size
        
        train_data = shuffled_pairs[:train_size]
        val_data = shuffled_pairs[train_size:train_size + val_size]
        test_data = shuffled_pairs[train_size + val_size:]
        
        # データコピー
        splits = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
        
        for split_name, data_list in splits.items():
            print(f"📦 {split_name}データ: {len(data_list)}個")
            
            for img_path, label_path in data_list:
                # 画像コピー
                dst_img = self.output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # ラベルコピー
                dst_label = self.output_dir / split_name / 'labels' / label_path.name
                shutil.copy2(label_path, dst_label)
        
        return len(train_data), len(val_data), len(test_data)
    
    def train_model(self, model_size='n', epochs=100, batch_size=16, patience=10):
        """YOLOv8モデル学習"""
        print(f"YOLOv8{model_size}モデル学習開始...")
        
        # 設定ファイル作成
        config_path = self.create_dataset_config()
        
        # YOLOモデル初期化
        model = YOLO(f'yolov8{model_size}.pt')
        
        # 学習実行
        results = model.train(
            data=str(config_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=patience,
            device='cpu',  # GPUが利用可能な場合は'0'に変更
            project=str(self.output_dir / 'models'),
            name='lesion_detection',
            exist_ok=True,
            verbose=True,
            plots=True,
            save_period=10
        )
        
        print("✅ 学習完了")
        return results
    
    def evaluate_model(self, model_path=None):
        """学習済みモデルの評価"""
        if model_path is None:
            # 最新の学習済みモデルを使用
            model_path = self.output_dir / 'models' / 'lesion_detection' / 'weights' / 'best.pt'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        print(f"モデル評価中: {model_path}")
        
        # モデル読み込み
        model = YOLO(str(model_path))
        
        # テストデータで評価
        test_images_dir = self.output_dir / 'test' / 'images'
        if not test_images_dir.exists() or not list(test_images_dir.glob('*')):
            print("テストデータが見つかりません。バリデーションデータで評価します。")
            test_images_dir = self.output_dir / 'val' / 'images'
        
        # 評価実行
        results = model.val(data=str(self.create_dataset_config()))
        
        print("📊 評価結果:")
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def test_detection(self, model_path=None, test_image_path=None):
        """個別画像での検出テスト"""
        if model_path is None:
            model_path = self.output_dir / 'models' / 'lesion_detection' / 'weights' / 'best.pt'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        # モデル読み込み
        model = YOLO(str(model_path))
        
        # テスト画像選択
        if test_image_path is None:
            test_images = list((self.output_dir / 'test' / 'images').glob('*'))
            if not test_images:
                test_images = list((self.output_dir / 'val' / 'images').glob('*'))
            if test_images:
                test_image_path = test_images[0]
        
        if test_image_path is None:
            print("テスト画像が見つかりません")
            return None
        
        print(f"🔍 テスト画像: {test_image_path}")
        
        # 推論実行
        results = model(str(test_image_path))
        
        # 結果表示
        for r in results:
            print(f"検出数: {len(r.boxes)} 個")
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.disease_classes[class_id]
                print(f"- {class_name}: {confidence:.3f}")
        
        # 結果画像保存
        output_path = self.output_dir / 'results' / f'detection_result_{Path(test_image_path).name}'
        results[0].save(str(output_path))
        print(f"💾 結果保存: {output_path}")
        
        return results
    
    def generate_report(self):
        """学習・評価レポート生成"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'classes': self.disease_classes,
                'total_classes': len(self.disease_classes)
            },
            'paths': {
                'images_dir': str(self.images_dir),
                'annotations_dir': str(self.annotations_dir),
                'output_dir': str(self.output_dir)
            }
        }
        
        # データセット統計
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split / 'images'
            if split_dir.exists():
                count = len(list(split_dir.glob('*')))
                report['dataset_info'][f'{split}_count'] = count
        
        # レポート保存
        report_path = self.output_dir / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 レポート保存: {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description='YOLOv8病変検出システム')
    parser.add_argument('images_dir', help='マスク済み画像ディレクトリ')
    parser.add_argument('-a', '--annotations', default='yolo_annotations', help='アノテーションディレクトリ')
    parser.add_argument('-o', '--output', default='yolo_dataset', help='出力ディレクトリ')
    parser.add_argument('-m', '--model', default='n', choices=['n', 's', 'm', 'l', 'x'], help='モデルサイズ')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='学習エポック数')
    parser.add_argument('--batch', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--eval-only', action='store_true', help='評価のみ実行')
    parser.add_argument('--test-only', action='store_true', help='テストのみ実行')
    
    args = parser.parse_args()
    
    # トレーナー初期化
    trainer = LesionYOLOTrainer(args.images_dir, args.annotations, args.output)
    
    if args.test_only:
        # テストのみ
        trainer.test_detection()
    elif args.eval_only:
        # 評価のみ
        trainer.evaluate_model()
    else:
        # 完全なパイプライン実行
        print("🚀 YOLOv8病変検出システム開始")
        
        # データセット準備
        train_count, val_count, test_count = trainer.prepare_dataset()
        print(f"データセット分割: train={train_count}, val={val_count}, test={test_count}")
        
        # モデル学習
        trainer.train_model(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch
        )
        
        # モデル評価
        trainer.evaluate_model()
        
        # テスト実行
        trainer.test_detection()
        
        # レポート生成
        trainer.generate_report()
        
        print("✅ 全処理完了")

if __name__ == '__main__':
    main()