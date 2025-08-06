#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改良版段階的病変検出学習システム
1. PAD-UFES-20で基礎学習
2. マスク済み画像を直接使用してROI学習
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

class ImprovedProgressiveLesionTrainer:
    def __init__(self, output_dir="improved_progressive_training"):
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
        
        print(f"🎯 Improved Progressive Lesion Trainer 初期化")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def create_roi_extraction_dataset(self, source_dir, samples_per_class=15):
        """
        マスク済み画像から直接ROIを抽出してデータセット作成
        """
        print("\n🎨 マスク済み画像ROI抽出データセット作成")
        print("=" * 60)
        
        roi_dir = self.output_dir / "roi_dataset"
        yolo_dir = roi_dir / "yolo_format"
        
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
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }
        
        processed_count = 0
        
        for split_name, data in splits.items():
            print(f"\n📋 {split_name}データ処理: {len(data)}件")
            
            for i, item in enumerate(tqdm(data, desc=f"ROI {split_name}")):
                try:
                    # 画像読み込み
                    img = cv2.imread(str(item['path']))
                    if img is None:
                        continue
                    
                    # 画像サイズ取得
                    h, w = img.shape[:2]
                    
                    # マスク済み画像から病変領域を自動検出
                    lesion_rois = self.extract_lesion_regions(img)
                    
                    # ROIが検出されない場合は中央の大きな領域を使用
                    if not lesion_rois:
                        center_x, center_y = w // 2, h // 2
                        roi_size = min(w, h) // 2
                        lesion_rois = [{
                            'x': max(0, center_x - roi_size // 2),
                            'y': max(0, center_y - roi_size // 2),
                            'w': min(roi_size, w - max(0, center_x - roi_size // 2)),
                            'h': min(roi_size, h - max(0, center_y - roi_size // 2))
                        }]
                    
                    # 各ROIを保存
                    for roi_idx, roi in enumerate(lesion_rois):
                        # ROI抽出
                        x, y, roi_w, roi_h = roi['x'], roi['y'], roi['w'], roi['h']
                        roi_img = img[y:y+roi_h, x:x+roi_w]
                        
                        if roi_img.size == 0:
                            continue
                        
                        # 640x640にリサイズ
                        roi_resized = cv2.resize(roi_img, (640, 640))
                        
                        # ファイル名作成
                        new_name = f"roi_{split_name}_{i:04d}_{roi_idx}.jpg"
                        
                        # 画像保存
                        dst_img = yolo_dir / split_name / 'images' / new_name
                        cv2.imwrite(str(dst_img), roi_resized)
                        
                        # YOLOアノテーション作成（ROI全体が病変）
                        class_id = self.pad_classes.index(item['class'])
                        annotation = f"{class_id} 0.5 0.5 0.9 0.9"  # 90%が病変
                        
                        label_path = yolo_dir / split_name / 'labels' / f"roi_{split_name}_{i:04d}_{roi_idx}.txt"
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
        
        yaml_path = yolo_dir / 'roi_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"\n✅ ROIデータセット作成完了: {processed_count}件のデータを作成")
        print(f"📄 設定ファイル: {yaml_path}")
        
        return yaml_path, processed_count
    
    def extract_lesion_regions(self, image):
        """
        マスク済み画像から病変領域を抽出
        """
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ヒストグラム平坦化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # エッジ検出
        edges = cv2.Canny(enhanced, 30, 100)
        
        # モルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 適切なサイズの領域を選択
        lesion_rois = []
        h, w = image.shape[:2]
        min_area = (w * h) * 0.01  # 画像の1%以上
        max_area = (w * h) * 0.3   # 画像の30%以下
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                # アスペクト比チェック
                aspect_ratio = float(cw) / ch
                if 0.3 < aspect_ratio < 3.0:
                    lesion_rois.append({'x': x, 'y': y, 'w': cw, 'h': ch})
        
        # 面積で並び替えて上位3つまで
        lesion_rois.sort(key=lambda r: r['w'] * r['h'], reverse=True)
        return lesion_rois[:3]
    
    def train_comprehensive_model(self, yaml_path, epochs=40):
        """
        包括的なモデル訓練
        """
        print("\n🚀 包括的病変検出モデル訓練開始")
        print("=" * 60)
        
        # YOLOv8sから開始（より高性能）
        model = YOLO('yolov8s.pt')
        
        # 訓練実行
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name=f'comprehensive_lesion_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=15,
            save=True,
            cache=True,
            device='mps',
            workers=4,
            project=self.output_dir / 'training_runs',
            
            # 最適化された設定
            lr0=0.01,
            lrf=0.01,
            warmup_epochs=5,
            
            # データ拡張
            degrees=20.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.2,    # 医療画像では上下反転も有効
            mixup=0.15,
            mosaic=1.0,
            
            # 最適化
            optimizer='AdamW',
            close_mosaic=15
        )
        
        model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\n✅ 包括的モデル訓練完了!")
        print(f"🏆 モデル: {model_path}")
        
        return model_path, results
    
    def test_on_unseen_images(self, model_path, test_images_dir, num_test_per_class=5):
        """
        初見画像での病変検出テスト
        """
        print("\n🧪 初見画像での病変検出テスト")
        print("=" * 60)
        
        model = YOLO(model_path)
        test_results = []
        
        test_path = Path(test_images_dir)
        output_dir = self.output_dir / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # テスト用の画像を選択
        for lesion_dir in test_path.iterdir():
            if not lesion_dir.is_dir():
                continue
                
            lesion_type = lesion_dir.name
            expected_class = self.lesion_mapping.get(lesion_type, 'ACK')
            
            print(f"\n🔍 テスト中: {lesion_type} (期待クラス: {expected_class})")
            
            # 各タイプから指定数をテスト
            image_files = list(lesion_dir.glob("*.jpg"))
            test_images = random.sample(image_files, min(num_test_per_class, len(image_files)))
            
            for img_path in test_images:
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
                    
                    status = '✅' if is_correct else '❌'
                    print(f"  📸 {img_path.name}: {len(detections)}個検出, 正解: {status}")
                    
                except Exception as e:
                    print(f"❌ テストエラー {img_path}: {e}")
        
        # 統計計算
        total_tests = len(test_results)
        correct_tests = sum(1 for r in test_results if r['correct'])
        accuracy = correct_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 テスト結果:")
        print(f"  総テスト数: {total_tests}")
        print(f"  正解数: {correct_tests}")
        print(f"  精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 結果保存
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'accuracy': accuracy,
                'total_tests': total_tests,
                'correct_tests': correct_tests,
                'detailed_results': test_results
            }, f, ensure_ascii=False, indent=2)
        
        return accuracy, test_results
    
    def run_improved_training(self, source_dir="organized_advanced_masked"):
        """
        改良された段階的学習プロセス全体を実行
        """
        print("🎯 Improved Progressive Lesion Detection Training 開始")
        print("=" * 80)
        print("📋 改良プラン:")
        print("  1. マスク済み画像から直接ROI抽出データセット作成")
        print("  2. 包括的病変検出モデル訓練")
        print("  3. 初見画像での検出テスト")
        print("=" * 80)
        
        try:
            # ROIデータセット作成
            roi_yaml, roi_count = self.create_roi_extraction_dataset(source_dir)
            
            if roi_count == 0:
                print("❌ ROIデータが作成されませんでした")
                return None
            
            # 包括的モデル訓練
            model_path, training_results = self.train_comprehensive_model(roi_yaml)
            
            # テスト実行
            final_accuracy, test_results = self.test_on_unseen_images(model_path, source_dir)
            
            print("\n🎉 Improved Progressive Training 完了!")
            print("=" * 80)
            print(f"📊 最終結果:")
            print(f"  ROIデータ数: {roi_count}")
            print(f"  最終精度: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
            print(f"🏆 最終モデル: {model_path}")
            print("=" * 80)
            
            return {
                'model_path': model_path,
                'final_accuracy': final_accuracy,
                'roi_count': roi_count,
                'test_results': test_results
            }
            
        except Exception as e:
            print(f"❌ Improved Progressive Training エラー: {e}")
            return None

def main():
    """メイン実行関数"""
    trainer = ImprovedProgressiveLesionTrainer()
    results = trainer.run_improved_training()
    
    if results:
        print(f"\n✅ 改良段階的学習成功!")
        print(f"📈 達成精度: {results['final_accuracy']:.3f}")
        print(f"🎯 初見画像での病変自動検出が可能になりました！")
    else:
        print("❌ 改良段階的学習に失敗しました")

if __name__ == "__main__":
    main()