#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAD-UFES-20モデルをYOLO抽出病変部で強化する追加学習システム
抽出された病変ROIを使用してモデルの精度を向上させる
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

class EnhancedPADLesionTrainer:
    def __init__(self, base_model_path=None, output_dir="enhanced_pad_training"):
        """
        強化学習システムの初期化
        """
        self.base_model_path = base_model_path or "runs/detect/pad_ufes_20_realistic_20250805_174734/weights/best.pt"
        self.output_dir = Path(output_dir)
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
        
        print(f"🎯 Enhanced PAD Lesion Trainer 初期化")
        print(f"📂 ベースモデル: {self.base_model_path}")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def extract_lesions_for_training(self, source_images_dir, confidence_threshold=0.3):
        """
        元画像から病変を抽出して訓練データを作成
        """
        print("🔍 YOLO病変抽出による訓練データ作成開始...")
        
        source_path = Path(source_images_dir)
        extraction_dir = self.output_dir / "extracted_lesions"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced PAD Lesion Extractorを使用
        from enhanced_pad_lesion_extractor import EnhancedPADLesionExtractor
        extractor = EnhancedPADLesionExtractor()
        
        extracted_data = []
        
        # 各病変タイプのディレクトリを処理
        for lesion_dir in source_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            mapped_type = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"📂 処理中: {lesion_type} -> {mapped_type}")
            
            # 各画像を処理（数を制限）
            image_files = list(lesion_dir.glob("*.jpg"))[:5]  # 各タイプから5枚に制限
            
            for img_path in tqdm(image_files, desc=f"Extracting {lesion_type}"):
                try:
                    # 病変抽出実行
                    lesions = extractor.extract_lesions_from_image(
                        img_path,
                        extraction_dir / lesion_type,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # 抽出結果を記録
                    for lesion in lesions:
                        extracted_data.append({
                            'original_image': str(img_path),
                            'lesion_image': lesion['filename'],
                            'true_class': mapped_type,
                            'predicted_class': lesion['class'],
                            'confidence': lesion['confidence'],
                            'bbox': lesion['bbox'],
                            'area': lesion['area']
                        })
                        
                except Exception as e:
                    print(f"❌ エラー処理 {img_path}: {e}")
        
        # 抽出結果保存
        extraction_summary = extraction_dir / "extraction_summary.json"
        with open(extraction_summary, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 病変抽出完了: {len(extracted_data)}個の病変ROIを抽出")
        return extracted_data, extraction_dir
    
    def create_enhanced_dataset(self, extracted_data, extraction_dir):
        """
        抽出された病変ROIからYOLO学習用データセットを作成
        """
        print("📊 拡張データセット作成中...")
        
        dataset_dir = self.output_dir / "enhanced_dataset"
        yolo_dir = dataset_dir / "yolo_format"
        
        # ディレクトリ構造作成
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # データを分割
        np.random.shuffle(extracted_data)
        total = len(extracted_data)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        splits = {
            'train': extracted_data[:train_end],
            'val': extracted_data[train_end:val_end],
            'test': extracted_data[val_end:]
        }
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\\n{split_name}データ処理中: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"Processing {split_name}")):
                try:
                    # 元の病変画像パス
                    lesion_img_name = item['lesion_image']
                    original_dir = Path(item['original_image']).parent.name
                    lesion_path = extraction_dir / original_dir / lesion_img_name
                    
                    if not lesion_path.exists():
                        continue
                    
                    # 新しいファイル名
                    class_name = item['true_class']
                    new_name = f"{class_name}_{split_name}_{i:04d}.jpg"
                    
                    # 画像をコピー
                    dst_img = yolo_dir / split_name / 'images' / new_name
                    shutil.copy2(lesion_path, dst_img)
                    
                    # YOLOアノテーション作成（病変ROI画像なので全体が病変）
                    class_id = self.pad_classes.index(class_name)
                    annotation = f"{class_id} 0.5 0.5 0.8 0.8"  # 中央80%が病変
                    
                    label_path = yolo_dir / split_name / 'labels' / f"{class_name}_{split_name}_{i:04d}.txt"
                    with open(label_path, 'w') as f:
                        f.write(annotation)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"❌ データ処理エラー: {e}")
        
        # YAML設定ファイル作成
        yaml_config = {
            'path': str(yolo_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.pad_classes),
            'names': self.pad_classes
        }
        
        yaml_path = yolo_dir / 'enhanced_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"✅ 拡張データセット作成完了: {processed_count}件のデータ")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def fine_tune_model(self, yaml_path, epochs=50, batch_size=16):
        """
        PAD-UFES-20モデルを拡張データセットでファインチューニング
        """
        print("🚀 PAD-UFES-20モデルのファインチューニング開始...")
        
        # ベースモデルをロード
        if Path(self.base_model_path).exists():
            model = YOLO(self.base_model_path)
            print(f"✅ ベースモデルロード: {self.base_model_path}")
        else:
            print("⚠️ ベースモデルが見つかりません。YOLOv8nから開始...")
            model = YOLO('yolov8n.pt')
        
        # ファインチューニング実行
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            name=f'enhanced_pad_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=10,
            save=True,
            cache=True,
            device='mps',
            workers=2,  # ワーカー数を減らす
            project=self.output_dir / 'training_runs',
            # 学習率を下げてファインチューニング
            lr0=0.001,  # 低い学習率
            lrf=0.001,
            warmup_epochs=5,
            # データ拡張を調整
            degrees=5.0,
            translate=0.05,
            scale=0.1,
            fliplr=0.5,
            mixup=0.1,
            # NMSタイムアウト対策
            agnostic_nms=True,
            max_det=100  # 最大検出数を制限
        )
        
        print("✅ ファインチューニング完了!")
        return results
    
    def evaluate_enhanced_model(self, model_path, test_images_dir):
        """
        強化されたモデルの性能を評価
        """
        print("📊 強化モデルの性能評価開始...")
        
        model = YOLO(model_path)
        test_results = []
        
        test_path = Path(test_images_dir)
        
        for lesion_dir in test_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            mapped_type = self.lesion_mapping.get(lesion_type, 'ACK')
            
            # テスト画像を処理
            image_files = list(lesion_dir.glob("*.jpg"))[:5]  # 各タイプから5枚テスト
            
            for img_path in image_files:
                results = model(img_path, conf=0.25, save=False)
                
                detected_classes = []
                confidences = []
                
                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = self.pad_classes[cls_id]
                            
                            detected_classes.append(class_name)
                            confidences.append(conf)
                
                test_results.append({
                    'image': img_path.name,
                    'true_class': mapped_type,
                    'detected_classes': detected_classes,
                    'confidences': confidences,
                    'correct': mapped_type in detected_classes if detected_classes else False
                })
        
        # 精度計算
        correct_predictions = sum(1 for r in test_results if r['correct'])
        total_predictions = len(test_results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"📈 評価結果:")
        print(f"  総テスト数: {total_predictions}")
        print(f"  正解数: {correct_predictions}")
        print(f"  精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 結果保存
        evaluation_path = self.output_dir / "evaluation_results.json"
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'detailed_results': test_results
            }, f, ensure_ascii=False, indent=2)
        
        return accuracy, test_results
    
    def run_complete_enhancement(self, source_images_dir, confidence_threshold=0.3, epochs=30):
        """
        完全な強化学習プロセスを実行
        """
        print("🎯 PAD-UFES-20モデル強化プロセス開始")
        print("=" * 60)
        
        # 1. 病変抽出
        extracted_data, extraction_dir = self.extract_lesions_for_training(
            source_images_dir, confidence_threshold
        )
        
        if len(extracted_data) == 0:
            print("❌ 抽出された病変データがありません")
            return None
        
        # 2. 拡張データセット作成
        yaml_path, data_count = self.create_enhanced_dataset(extracted_data, extraction_dir)
        
        if data_count == 0:
            print("❌ 処理されたデータがありません")
            return None
        
        # 3. ファインチューニング実行
        training_results = self.fine_tune_model(yaml_path, epochs=epochs)
        
        # 4. 最良モデルのパス取得
        best_model_path = training_results.save_dir / 'weights' / 'best.pt'
        
        # 5. 性能評価
        accuracy, test_results = self.evaluate_enhanced_model(
            best_model_path, source_images_dir
        )
        
        print(f"\\n🎉 PAD-UFES-20モデル強化完了!")
        print(f"📊 最終精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"🏆 強化モデル: {best_model_path}")
        
        return {
            'model_path': best_model_path,
            'accuracy': accuracy,
            'data_count': data_count,
            'training_results': training_results
        }

def main():
    """メイン実行関数"""
    print("🎯 Enhanced PAD-UFES-20 Lesion Trainer")
    print("=" * 50)
    
    # 既存のPAD-UFES-20モデルを使用
    base_model = "runs/detect/pad_ufes_20_realistic_20250805_174734/weights/best.pt"
    
    trainer = EnhancedPADLesionTrainer(base_model_path=base_model)
    
    # 強化学習実行
    results = trainer.run_complete_enhancement(
        source_images_dir="organized_advanced_masked",
        confidence_threshold=0.4,  # 高品質な病変のみ使用
        epochs=25
    )
    
    if results:
        print(f"\\n✅ 強化学習成功!")
        print(f"📈 精度向上: {results['accuracy']:.3f}")
        print(f"💾 モデル保存: {results['model_path']}")
    else:
        print("❌ 強化学習に失敗しました")

if __name__ == "__main__":
    main()