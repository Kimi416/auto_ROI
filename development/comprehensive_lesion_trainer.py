#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
包括的病変検出学習システム
全1,047枚のマスク済み画像を使用した完全学習
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

class ComprehensiveLesionTrainer:
    def __init__(self, output_dir="comprehensive_lesion_training"):
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
        
        print(f"🎯 包括的病変検出学習システム初期化")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def create_comprehensive_dataset(self, source_dir="organized_advanced_masked"):
        """
        全てのマスク済み画像を使用したデータセット作成
        """
        print("\n📊 包括的データセット作成（全画像使用）")
        print("=" * 60)
        
        dataset_dir = self.output_dir / "comprehensive_dataset"
        yolo_dir = dataset_dir / "yolo_format"
        
        # ディレクトリ構造作成
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        source_path = Path(source_dir)
        all_data = []
        
        # 各病変タイプから全ての画像を収集
        for lesion_dir in source_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            mapped_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            # 全ての画像ファイルを取得
            image_files = list(lesion_dir.glob("*.jpg"))
            print(f"📂 収集中: {lesion_type} → {mapped_class} ({len(image_files)}枚)")
            
            for img_path in image_files:
                all_data.append({
                    'path': img_path,
                    'class': mapped_class,
                    'original_type': lesion_type
                })
        
        print(f"\n📊 総収集画像数: {len(all_data)}枚")
        
        # データシャッフルと分割（7:2:1）
        random.shuffle(all_data)
        total = len(all_data)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }
        
        print(f"📋 データ分割:")
        print(f"  Train: {len(splits['train'])}枚")
        print(f"  Val: {len(splits['val'])}枚")
        print(f"  Test: {len(splits['test'])}枚")
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\n🔄 {split_name}データ処理中: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"Processing {split_name}")):
                try:
                    # 画像読み込み
                    img = cv2.imread(str(item['path']))
                    if img is None:
                        continue
                    
                    # 画像リサイズ
                    img_resized = cv2.resize(img, (640, 640))
                    
                    # ファイル名作成
                    new_name = f"{item['class']}_{split_name}_{i:05d}.jpg"
                    
                    # 画像保存
                    dst_img = yolo_dir / split_name / 'images' / new_name
                    cv2.imwrite(str(dst_img), img_resized)
                    
                    # YOLOアノテーション作成（中央の80%に病変があると仮定）
                    class_id = self.pad_classes.index(item['class'])
                    annotation = f"{class_id} 0.5 0.5 0.8 0.8"
                    
                    label_path = yolo_dir / split_name / 'labels' / f"{item['class']}_{split_name}_{i:05d}.txt"
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
            'test': 'test/images',
            'nc': len(self.pad_classes),
            'names': self.pad_classes
        }
        
        yaml_path = yolo_dir / 'comprehensive_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"\n✅ 包括的データセット作成完了: {processed_count}件")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def train_comprehensive_model(self, yaml_path, epochs=30):
        """
        包括的モデル訓練（全データ使用）
        """
        print("\n🚀 包括的モデル訓練開始")
        print("=" * 60)
        
        # YOLOv8sで高性能訓練
        model = YOLO('yolov8s.pt')
        
        # 包括的設定で訓練実行
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=f'comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=10,
            save=True,
            cache=False,  # メモリ節約
            device='mps',
            workers=4,
            project=self.output_dir / 'training_runs',
            
            # 最適化された設定
            lr0=0.01,
            lrf=0.01,
            warmup_epochs=3,
            
            # データ拡張
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.1,
            mixup=0.1,
            mosaic=1.0,
            
            # 最適化
            optimizer='AdamW',
            close_mosaic=10
        )
        
        model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\n✅ 包括的モデル訓練完了!")
        print(f"🏆 モデル: {model_path}")
        
        return model_path, results
    
    def comprehensive_test(self, model_path, test_images_dir, num_test=5):
        """
        包括的テスト
        """
        print("\n🧪 包括的テスト開始")
        print("=" * 60)
        
        model = YOLO(model_path)
        test_results = []
        
        test_path = Path(test_images_dir)
        output_dir = self.output_dir / "comprehensive_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各病変タイプからテスト
        for lesion_dir in test_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            expected_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"\n🔍 テスト中: {lesion_type} (期待: {expected_class})")
            
            # 各タイプから指定数をテスト
            image_files = list(lesion_dir.glob("*.jpg"))[:num_test]
            
            for img_path in image_files:
                try:
                    # 推論実行
                    results = model(str(img_path), conf=0.2, save=True, save_dir=output_dir)
                    
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
        
        print(f"\n📊 包括的テスト結果:")
        print(f"  総テスト数: {total_tests}")
        print(f"  正解数: {correct_tests}")
        print(f"  精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy, test_results
    
    def run_comprehensive_training(self, source_dir="organized_advanced_masked"):
        """
        包括的学習プロセス全体を実行
        """
        print("🎯 包括的病変検出学習開始")
        print("=" * 80)
        print("📋 包括的プラン:")
        print("  1. 全マスク済み画像（1,047枚）でデータセット作成")
        print("  2. 包括的モデル訓練 (30エポック)")
        print("  3. 包括的テスト")
        print("=" * 80)
        
        try:
            # 包括的データセット作成
            dataset_yaml, data_count = self.create_comprehensive_dataset(source_dir)
            
            if data_count == 0:
                print("❌ データが作成されませんでした")
                return None
            
            # 包括的モデル訓練
            model_path, training_results = self.train_comprehensive_model(dataset_yaml)
            
            # 包括的テスト実行
            final_accuracy, test_results = self.comprehensive_test(model_path, source_dir)
            
            print("\n🎉 包括的学習完了!")
            print("=" * 80)
            print(f"📊 最終結果:")
            print(f"  使用データ数: {data_count}")
            print(f"  最終精度: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
            print(f"🏆 最終モデル: {model_path}")
            print("=" * 80)
            
            return {
                'model_path': model_path,
                'final_accuracy': final_accuracy,
                'data_count': data_count,
                'test_results': test_results
            }
            
        except Exception as e:
            print(f"❌ 包括的学習エラー: {e}")
            return None

def main():
    """メイン実行関数"""
    trainer = ComprehensiveLesionTrainer()
    results = trainer.run_comprehensive_training()
    
    if results:
        print(f"\n✅ 包括的学習成功!")
        print(f"📈 達成精度: {results['final_accuracy']:.3f}")
        print(f"🎯 全マスク済み画像を使用した病変自動検出が完成しました！")
        print(f"📊 データ改善: 80枚 → {results['data_count']}枚 ({results['data_count']/80:.1f}倍)")
    else:
        print("❌ 包括的学習に失敗しました")

if __name__ == "__main__":
    main()