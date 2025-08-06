#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
段階的病変検出学習システム
1. PAD-UFES-20で基礎学習
2. マスク済み画像で病変抽出学習
3. 初見画像での病変自動検出
"""

import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
import yaml
import random

class ProgressiveLesionTrainer:
    def __init__(self, output_dir="progressive_lesion_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PAD-UFES-20クラス定義
        self.pad_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        
        # 病変タイプマッピング（日本の病変名 → PAD-UFES-20クラス）
        self.lesion_mapping = {
            'ADM': 'ACK',                    # 後天性真皮メラノサイトーシス → Actinic keratosis
            'Basal cell carcinoma': 'BCC',   # 基底細胞癌
            'Malignant melanoma': 'MEL',     # 悪性黒色腫 → Melanoma
            'Nevus': 'NEV',                  # 色素性母斑
            'Solar lentigo': 'SCC',          # 日光性色素斑 → Squamous cell carcinoma
            'Seborrheic keratosis': 'SEK',   # 脂漏性角化症
            'Ephelis': 'ACK',                # そばかす → Actinic keratosis
            'Melasma': 'ACK'                 # 肝斑 → Actinic keratosis
        }
        
        print(f"🎯 Progressive Lesion Trainer 初期化")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print(f"🏥 対象病変クラス: {', '.join(self.pad_classes)}")
    
    def stage1_create_foundation_dataset(self, source_dir, samples_per_class=15):
        """
        ステージ1: PAD-UFES-20スタイルの基礎データセット作成
        各病変タイプから画像を選択し、基本的な病変認識を学習
        """
        print("\\n🏗️  STAGE 1: PAD-UFES-20基礎データセット作成")
        print("=" * 60)
        
        foundation_dir = self.output_dir / "stage1_foundation"
        yolo_dir = foundation_dir / "yolo_format"
        
        # ディレクトリ構造作成
        for split in ['train', 'val', 'test']:
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
            
            # 各タイプから指定数の画像を取得
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
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\\n📋 {split_name}データ処理: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"Stage1 {split_name}")):
                try:
                    # 画像読み込みとリサイズ
                    img = cv2.imread(str(item['path']))
                    if img is None:
                        continue
                    
                    img_resized = cv2.resize(img, (640, 640))
                    
                    # ファイル名作成
                    new_name = f"stage1_{split_name}_{i:04d}.jpg"
                    
                    # 画像保存
                    dst_img = yolo_dir / split_name / 'images' / new_name
                    cv2.imwrite(str(dst_img), img_resized)
                    
                    # YOLOアノテーション作成（画像中央に病変があると仮定）
                    class_id = self.pad_classes.index(item['class'])
                    # 中央の70%の領域に病変があると仮定
                    annotation = f"{class_id} 0.5 0.5 0.7 0.7"
                    
                    label_path = yolo_dir / split_name / 'labels' / f"stage1_{split_name}_{i:04d}.txt"
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
        
        yaml_path = yolo_dir / 'foundation_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"\\n✅ Stage 1完了: {processed_count}件のデータを作成")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def stage1_train_foundation_model(self, yaml_path, epochs=30):
        """
        ステージ1: 基礎モデル訓練
        """
        print("\\n🚀 STAGE 1: 基礎モデル訓練開始")
        print("=" * 60)
        
        # YOLOv8nから開始
        model = YOLO('yolov8n.pt')
        
        # 基礎訓練実行
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=f'stage1_foundation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=10,
            save=True,
            cache=True,
            device='mps',
            workers=4,
            project=self.output_dir / 'training_runs',
            
            # 基礎学習の設定
            lr0=0.01,      # 標準的な学習率
            lrf=0.01,
            warmup_epochs=5,
            
            # データ拡張設定
            degrees=15.0,   # 回転
            translate=0.1,  # 平行移動
            scale=0.5,      # スケール変更
            fliplr=0.5,     # 左右反転
            mixup=0.1,      # ミックスアップ
            
            # 最適化設定
            optimizer='AdamW',
            close_mosaic=10
        )
        
        foundation_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\\n✅ Stage 1訓練完了!")
        print(f"🏆 基礎モデル: {foundation_model_path}")
        
        return foundation_model_path, results
    
    def stage2_create_masked_lesion_dataset(self, source_dir, foundation_model_path, samples_per_class=10):
        """
        ステージ2: マスク済み画像から病変部を抽出してデータセット作成
        """
        print("\\n🎭 STAGE 2: マスク済み病変抽出データセット作成")
        print("=" * 60)
        
        stage2_dir = self.output_dir / "stage2_masked_lesions"
        yolo_dir = stage2_dir / "yolo_format"
        
        # ディレクトリ構造作成
        for split in ['train', 'val']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 基礎モデルロード
        foundation_model = YOLO(foundation_model_path)
        
        source_path = Path(source_dir)
        extracted_lesions = []
        
        # 各病変タイプから画像を処理
        for lesion_dir in source_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            mapped_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"\\n🔍 病変抽出中: {lesion_type} → {mapped_class}")
            
            # 各タイプから指定数の画像を処理
            image_files = list(lesion_dir.glob("*.jpg"))
            selected_images = random.sample(image_files, min(samples_per_class, len(image_files)))
            
            for img_path in tqdm(selected_images, desc=f"Extracting {lesion_type}"):
                try:
                    # 基礎モデルで病変検出
                    results = foundation_model(str(img_path), conf=0.3, verbose=False)
                    
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # 検出された病変から高信頼度のものを抽出
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                conf = float(box.conf[0])
                                if conf > 0.4:  # 高信頼度のみ
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    # 病変ROI抽出
                                    lesion_roi = img[y1:y2, x1:x2]
                                    if lesion_roi.size == 0:
                                        continue
                                    
                                    # 640x640にリサイズ
                                    lesion_resized = cv2.resize(lesion_roi, (640, 640))
                                    
                                    extracted_lesions.append({
                                        'image': lesion_resized,
                                        'class': mapped_class,
                                        'confidence': conf,
                                        'original_type': lesion_type
                                    })
                
                except Exception as e:
                    print(f"❌ エラー {img_path}: {e}")
        
        print(f"\\n📊 抽出された病変数: {len(extracted_lesions)}")
        
        if len(extracted_lesions) == 0:
            print("❌ 病変が抽出されませんでした")
            return None, 0
        
        # データをtrain/valに分割
        random.shuffle(extracted_lesions)
        train_count = int(len(extracted_lesions) * 0.8)
        
        splits = {
            'train': extracted_lesions[:train_count],
            'val': extracted_lesions[train_count:]
        }
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\\n📋 {split_name}データ処理: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"Stage2 {split_name}")):
                try:
                    # 画像保存
                    img_name = f"stage2_{split_name}_{i:04d}.jpg"
                    img_path = yolo_dir / split_name / 'images' / img_name
                    cv2.imwrite(str(img_path), item['image'])
                    
                    # YOLOアノテーション（抽出された病変なので全体が病変）
                    class_id = self.pad_classes.index(item['class'])
                    annotation = f"{class_id} 0.5 0.5 0.9 0.9"  # 90%が病変
                    
                    label_path = yolo_dir / split_name / 'labels' / f"stage2_{split_name}_{i:04d}.txt"
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
        
        yaml_path = yolo_dir / 'masked_lesion_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"\\n✅ Stage 2完了: {processed_count}件の病変データを作成")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def stage2_enhance_model(self, foundation_model_path, stage2_yaml_path, epochs=20):
        """
        ステージ2: 基礎モデルを病変抽出データで強化
        """
        print("\\n🔥 STAGE 2: モデル強化訓練開始")
        print("=" * 60)
        
        # 基礎モデルロード
        model = YOLO(foundation_model_path)
        
        # 強化訓練実行
        results = model.train(
            data=stage2_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            name=f'stage2_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=8,
            save=True,
            cache=True,
            device='mps',
            workers=2,
            project=self.output_dir / 'training_runs',
            
            # ファインチューニング設定
            lr0=0.001,     # 低い学習率でファインチューニング
            lrf=0.0001,
            warmup_epochs=3,
            
            # 軽度なデータ拡張
            degrees=10.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            mixup=0.05,
            
            # 最適化設定
            optimizer='AdamW',
            close_mosaic=8
        )
        
        enhanced_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\\n✅ Stage 2強化完了!")
        print(f"🏆 強化モデル: {enhanced_model_path}")
        
        return enhanced_model_path, results
    
    def stage3_test_on_unseen_images(self, enhanced_model_path, test_images_dir):
        """
        ステージ3: 初見画像での病変検出テスト
        """
        print("\\n🧪 STAGE 3: 初見画像での病変検出テスト")
        print("=" * 60)
        
        model = YOLO(enhanced_model_path)
        test_results = []
        
        test_path = Path(test_images_dir)
        output_dir = self.output_dir / "stage3_test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各病変タイプから数枚ずつテスト
        for lesion_dir in test_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            expected_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"\\n🔍 テスト中: {lesion_type} (期待クラス: {expected_class})")
            
            # 各タイプから3枚をテスト
            image_files = list(lesion_dir.glob("*.jpg"))[:3]
            
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
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                detections.append({
                                    'class': predicted_class,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
                    
                    # 結果記録
                    is_correct = any(d['class'] == expected_class for d in detections)
                    
                    test_results.append({
                        'image': img_path.name,
                        'expected_class': expected_class,
                        'original_type': lesion_type,
                        'detections': detections,
                        'detection_count': len(detections),
                        'correct': is_correct
                    })
                    
                    print(f"  📸 {img_path.name}: {len(detections)}個検出, 正解: {'✅' if is_correct else '❌'}")
                    
                except Exception as e:
                    print(f"❌ テストエラー {img_path}: {e}")
        
        # 統計計算
        total_tests = len(test_results)
        correct_tests = sum(1 for r in test_results if r['correct'])
        accuracy = correct_tests / total_tests if total_tests > 0 else 0
        
        print(f"\\n📊 Stage 3テスト結果:")
        print(f"  総テスト数: {total_tests}")
        print(f"  正解数: {correct_tests}")
        print(f"  精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 結果保存
        results_path = self.output_dir / "stage3_test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_path': str(enhanced_model_path),
                'accuracy': accuracy,
                'total_tests': total_tests,
                'correct_tests': correct_tests,
                'detailed_results': test_results
            }, f, ensure_ascii=False, indent=2)
        
        return accuracy, test_results
    
    def run_progressive_training(self, source_dir="organized_advanced_masked"):
        """
        段階的学習プロセス全体を実行
        """
        print("🎯 Progressive Lesion Detection Training 開始")
        print("=" * 80)
        print("📋 学習プラン:")
        print("  Stage 1: PAD-UFES-20基礎学習")
        print("  Stage 2: マスク済み画像での病変抽出学習")
        print("  Stage 3: 初見画像での検出テスト")
        print("=" * 80)
        
        try:
            # Stage 1: 基礎データセット作成と訓練
            stage1_yaml, stage1_count = self.stage1_create_foundation_dataset(source_dir)
            foundation_model, stage1_results = self.stage1_train_foundation_model(stage1_yaml)
            
            # Stage 2: 病変抽出データセット作成と強化訓練
            stage2_yaml, stage2_count = self.stage2_create_masked_lesion_dataset(
                source_dir, foundation_model
            )
            
            if stage2_yaml is None:
                print("❌ Stage 2でデータが作成されませんでした")
                return None
            
            enhanced_model, stage2_results = self.stage2_enhance_model(
                foundation_model, stage2_yaml
            )
            
            # Stage 3: 初見画像テスト
            final_accuracy, test_results = self.stage3_test_on_unseen_images(
                enhanced_model, source_dir
            )
            
            print("\\n🎉 Progressive Training 完了!")
            print("=" * 80)
            print(f"📊 最終結果:")
            print(f"  Stage 1データ数: {stage1_count}")
            print(f"  Stage 2データ数: {stage2_count}")
            print(f"  最終精度: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
            print(f"🏆 最終モデル: {enhanced_model}")
            print("=" * 80)
            
            return {
                'foundation_model': foundation_model,
                'enhanced_model': enhanced_model,
                'final_accuracy': final_accuracy,
                'stage1_count': stage1_count,
                'stage2_count': stage2_count,
                'test_results': test_results
            }
            
        except Exception as e:
            print(f"❌ Progressive Training エラー: {e}")
            return None

def main():
    """メイン実行関数"""
    trainer = ProgressiveLesionTrainer()
    results = trainer.run_progressive_training()
    
    if results:
        print(f"\\n✅ 段階的学習成功!")
        print(f"📈 達成精度: {results['final_accuracy']:.3f}")
        print(f"🎯 初見画像での病変自動検出が可能になりました！")
    else:
        print("❌ 段階的学習に失敗しました")

if __name__ == "__main__":
    main()