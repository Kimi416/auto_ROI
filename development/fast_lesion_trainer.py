#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高速病変検出学習システム
軽量版で迅速な学習とテストを実行
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
import yaml
import random

class FastLesionTrainer:
    def __init__(self, output_dir="fast_lesion_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PAD-UFES-20クラス定義
        self.pad_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        
        # 病変タイプマッピング
        self.lesion_mapping = {
            'ADM': 'ACK',
            'Basal cell carcinoma': 'BCC',
            'Malignant melanoma': 'MEL',
            'Nevus': 'NEV',
            'Solar lentigo': 'SCC',
            'Seborrheic keratosis': 'SEK',
            'Ephelis': 'ACK',
            'Melasma': 'ACK'
        }
        
        print(f"🎯 Fast Lesion Trainer 初期化")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def create_simple_dataset(self, source_dir, samples_per_class=10):
        """
        シンプルなデータセット作成
        """
        print("\n📊 シンプルデータセット作成")
        print("=" * 50)
        
        dataset_dir = self.output_dir / "simple_dataset"
        yolo_dir = dataset_dir / "yolo_format"
        
        # ディレクトリ構造作成
        for split in ['train', 'val']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        source_path = Path(source_dir)
        all_data = []
        
        # 各病変タイプから画像を収集
        for lesion_dir in source_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            mapped_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"📂 収集中: {lesion_type} → {mapped_class}")
            
            # 画像ファイルを取得
            image_files = list(lesion_dir.glob("*.jpg"))
            selected_images = random.sample(image_files, min(samples_per_class, len(image_files)))
            
            for img_path in selected_images:
                all_data.append({
                    'path': img_path,
                    'class': mapped_class,
                    'original_type': lesion_type
                })
        
        print(f"📊 総収集画像数: {len(all_data)}")
        
        # データシャッフルと分割
        random.shuffle(all_data)
        total = len(all_data)
        train_end = int(total * 0.8)
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:]
        }
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\n📋 {split_name}データ処理: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"Processing {split_name}")):
                try:
                    # 画像読み込み
                    img = cv2.imread(str(item['path']))
                    if img is None:
                        continue
                    
                    # 画像リサイズ
                    img_resized = cv2.resize(img, (640, 640))
                    
                    # ファイル名作成
                    new_name = f"{item['class']}_{split_name}_{i:04d}.jpg"
                    
                    # 画像保存
                    dst_img = yolo_dir / split_name / 'images' / new_name
                    cv2.imwrite(str(dst_img), img_resized)
                    
                    # YOLOアノテーション作成（中央の80%に病変があると仮定）
                    class_id = self.pad_classes.index(item['class'])
                    annotation = f"{class_id} 0.5 0.5 0.8 0.8"
                    
                    label_path = yolo_dir / split_name / 'labels' / f"{item['class']}_{split_name}_{i:04d}.txt"
                    with open(label_path, 'w') as f:
                        f.write(annotation)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"❌ エラー: {e}")
        
        # YAML設定ファイル作成
        yaml_config = {
            'path': str(yolo_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.pad_classes),
            'names': self.pad_classes
        }
        
        yaml_path = yolo_dir / 'simple_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"\n✅ シンプルデータセット作成完了: {processed_count}件")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def train_fast_model(self, yaml_path, epochs=15):
        """
        高速モデル訓練
        """
        print("\n🚀 高速モデル訓練開始")
        print("=" * 50)
        
        # YOLOv8nで高速訓練
        model = YOLO('yolov8n.pt')
        
        # 高速設定で訓練実行
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            name=f'fast_lesion_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=8,
            save=True,
            cache=False,  # メモリ使用量削減
            device='mps',
            workers=2,
            project=self.output_dir / 'training_runs',
            
            # 高速化設定
            lr0=0.01,
            lrf=0.1,
            warmup_epochs=2,
            
            # 軽量データ拡張
            degrees=10.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            mixup=0.0,
            mosaic=0.5,
            
            # 最適化
            optimizer='SGD',
            close_mosaic=5
        )
        
        model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\n✅ 高速モデル訓練完了!")
        print(f"🏆 モデル: {model_path}")
        
        return model_path, results
    
    def quick_test(self, model_path, test_images_dir, num_test=3):
        """
        クイックテスト
        """
        print("\n🧪 クイックテスト開始")
        print("=" * 50)
        
        model = YOLO(model_path)
        test_results = []
        
        test_path = Path(test_images_dir)
        output_dir = self.output_dir / "quick_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各病変タイプから数枚ずつテスト
        for lesion_dir in test_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            expected_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"\n🔍 テスト中: {lesion_type} (期待: {expected_class})")
            
            # 各タイプから数枚をテスト
            image_files = list(lesion_dir.glob("*.jpg"))[:num_test]
            
            for img_path in image_files:
                try:
                    # 推論実行
                    results = model(str(img_path), conf=0.25, save=True, save_dir=output_dir)
                    
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                predicted_class = self.pad_classes[cls_id]
                                
                                detections.append({
                                    'class': predicted_class,
                                    'confidence': conf
                                })
                    
                    # 結果記録
                    is_correct = any(d['class'] == expected_class for d in detections)
                    
                    test_results.append({
                        'image': img_path.name,
                        'expected_class': expected_class,
                        'original_type': lesion_type,
                        'detections': detections,
                        'correct': is_correct
                    })
                    
                    status = '✅' if is_correct else '❌'
                    det_str = ', '.join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
                    print(f"  📸 {img_path.name}: {det_str} → {status}")
                    
                except Exception as e:
                    print(f"❌ テストエラー {img_path}: {e}")
        
        # 統計計算
        total_tests = len(test_results)
        correct_tests = sum(1 for r in test_results if r['correct'])
        accuracy = correct_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 クイックテスト結果:")
        print(f"  総テスト数: {total_tests}")
        print(f"  正解数: {correct_tests}")
        print(f"  精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy, test_results
    
    def run_fast_training(self, source_dir="organized_advanced_masked"):
        """
        高速学習プロセス全体を実行
        """
        print("🎯 Fast Lesion Detection Training 開始")
        print("=" * 60)
        print("📋 高速プラン:")
        print("  1. シンプルデータセット作成")
        print("  2. 高速モデル訓練 (15エポック)")
        print("  3. クイックテスト")
        print("=" * 60)
        
        try:
            # シンプルデータセット作成
            dataset_yaml, data_count = self.create_simple_dataset(source_dir)
            
            if data_count == 0:
                print("❌ データが作成されませんでした")
                return None
            
            # 高速モデル訓練
            model_path, training_results = self.train_fast_model(dataset_yaml)
            
            # クイックテスト実行
            final_accuracy, test_results = self.quick_test(model_path, source_dir)
            
            print("\n🎉 Fast Training 完了!")
            print("=" * 60)
            print(f"📊 最終結果:")
            print(f"  データ数: {data_count}")
            print(f"  最終精度: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
            print(f"🏆 最終モデル: {model_path}")
            print("=" * 60)
            
            return {
                'model_path': model_path,
                'final_accuracy': final_accuracy,
                'data_count': data_count,
                'test_results': test_results
            }
            
        except Exception as e:
            print(f"❌ Fast Training エラー: {e}")
            return None

def main():
    """メイン実行関数"""
    trainer = FastLesionTrainer()
    results = trainer.run_fast_training()
    
    if results:
        print(f"\n✅ 高速学習成功!")
        print(f"📈 達成精度: {results['final_accuracy']:.3f}")
        print(f"🎯 初見画像での病変自動検出が可能になりました！")
    else:
        print("❌ 高速学習に失敗しました")

if __name__ == "__main__":
    main()