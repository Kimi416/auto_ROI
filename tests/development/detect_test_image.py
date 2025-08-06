#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test.jpgに対してoptimal_stable_phase2モデルを使用して病変を検出・抽出するスクリプト
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json

class TestImageDetector:
    """test.jpg専用の病変検出クラス"""
    
    def __init__(self):
        """最適化されたモデルを自動で読み込み"""
        # 最高性能のモデルパスを設定
        self.model_path = "/Users/iinuma/Desktop/自動ROI/runs/detect/optimal_stable_phase2/weights/best.pt"
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
            
        print(f"モデルを読み込み中: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        print(f"検出可能なクラス: {list(self.class_names.values())}")
        
    def detect_lesions_in_test_image(self, conf_threshold=0.25):
        """
        test.jpgから病変を検出して抽出
        
        Args:
            conf_threshold: 信頼度の閾値
        
        Returns:
            検出結果のリスト
        """
        image_path = "/Users/iinuma/Desktop/自動ROI/test.jpg"
        output_dir = "/Users/iinuma/Desktop/自動ROI/test_detection_results"
        
        # test.jpgの存在確認
        if not Path(image_path).exists():
            raise FileNotFoundError(f"test.jpgが見つかりません: {image_path}")
        
        # 画像を読み込み
        print(f"画像を読み込み中: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("画像の読み込みに失敗しました")
        
        print(f"画像サイズ: {image.shape}")
        
        # 検出を実行
        print("病変検出を実行中...")
        results = self.model(image, conf=conf_threshold, iou=0.45)
        
        # 出力ディレクトリを作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        detections = []
        
        # 各検出結果を処理
        for r in results:
            boxes = r.boxes
            if boxes is None:
                print("検出された病変はありません")
                continue
                
            print(f"検出された病変数: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[cls]
                
                print(f"病変 {i+1}: {class_name} (信頼度: {conf:.3f})")
                print(f"  座標: ({x1}, {y1}) - ({x2}, {y2})")
                
                # 病変部分を切り抜き（マージンを追加）
                margin = 20
                y1_crop = max(0, y1 - margin)
                y2_crop = min(image.shape[0], y2 + margin)
                x1_crop = max(0, x1 - margin)
                x2_crop = min(image.shape[1], x2 + margin)
                
                lesion_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # ファイル名を生成
                crop_filename = f"test_lesion_{i+1}_{class_name}_{conf:.3f}.jpg"
                crop_path = output_path / crop_filename
                
                # 切り抜いた画像を保存
                cv2.imwrite(str(crop_path), lesion_crop)
                print(f"  保存: {crop_path}")
                
                # 検出結果を記録
                detection = {
                    'lesion_id': i + 1,
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'bbox_with_margin': [int(x1_crop), int(y1_crop), int(x2_crop), int(y2_crop)],
                    'crop_path': str(crop_path),
                    'area': int((x2 - x1) * (y2 - y1))
                }
                detections.append(detection)
        
        # 検出結果を画像に描画
        print("検出結果を画像に描画中...")
        annotated_image = self.draw_detections(image, results)
        annotated_path = output_path / "test_detection_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_image)
        print(f"アノテーション画像を保存: {annotated_path}")
        
        # 検出結果をJSONで保存
        results_path = output_path / 'test_detection_info.json'
        detection_info = {
            'model_path': self.model_path,
            'image_path': image_path,
            'image_size': list(image.shape),
            'conf_threshold': conf_threshold,
            'total_detections': len(detections),
            'detections': detections
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detection_info, f, indent=2, ensure_ascii=False)
        print(f"検出情報を保存: {results_path}")
        
        return detections
    
    def draw_detections(self, image, results):
        """検出結果を画像に描画"""
        annotated = image.copy()
        
        # 病変クラスごとの色設定
        colors = {
            'ADM': (255, 0, 0),                    # 赤
            'Ephelis': (0, 255, 0),               # 緑  
            'Melasma': (0, 0, 255),               # 青
            'Solar lentigo': (255, 255, 0),       # 黄
            'Nevus': (255, 0, 255),               # マゼンタ
            'Basal cell carcinoma': (0, 255, 255), # シアン
            'Seborrheic keratosis': (128, 0, 128), # 紫
            'Malignant melanoma': (255, 128, 0),   # オレンジ
        }
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[cls]
                
                # 色を取得（デフォルトは白）
                color = colors.get(class_name, (255, 255, 255))
                
                # バウンディングボックスを描画
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # ラベルを描画
                label = f"{i+1}: {class_name} ({conf:.3f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # ラベル背景を描画
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8), 
                            (x1 + label_size[0] + 4, y1), color, -1)
                
                # ラベルテキストを描画
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def print_summary(self, detections):
        """検出結果のサマリーを表示"""
        print("\n" + "="*50)
        print("🔍 test.jpg 病変検出結果サマリー")
        print("="*50)
        
        if not detections:
            print("❌ 病変は検出されませんでした")
            print(f"   - 信頼度閾値を下げて再試行してください")
            return
        
        print(f"✅ 検出された病変数: {len(detections)}")
        
        # クラス別の統計
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\n📊 病変タイプ別の検出数:")
        for cls, count in sorted(class_counts.items()):
            print(f"   • {cls}: {count}個")
        
        # 信頼度の統計
        confidences = [det['confidence'] for det in detections]
        print(f"\n📈 信頼度の統計:")
        print(f"   • 最高: {max(confidences):.3f}")
        print(f"   • 最低: {min(confidences):.3f}")
        print(f"   • 平均: {np.mean(confidences):.3f}")
        
        # 面積の統計
        areas = [det['area'] for det in detections]
        print(f"\n📏 病変面積の統計 (ピクセル²):")
        print(f"   • 最大: {max(areas):,}")
        print(f"   • 最小: {min(areas):,}")
        print(f"   • 平均: {int(np.mean(areas)):,}")
        
        print(f"\n💾 結果ファイル:")
        print(f"   • 切り抜き画像: test_detection_results/")
        print(f"   • アノテーション画像: test_detection_results/test_detection_annotated.jpg")
        print(f"   • 詳細情報: test_detection_results/test_detection_info.json")
        print("="*50)

def main():
    """メイン実行関数"""
    try:
        # 検出器を初期化
        detector = TestImageDetector()
        
        # test.jpgから病変を検出
        detections = detector.detect_lesions_in_test_image(conf_threshold=0.25)
        
        # 結果のサマリーを表示
        detector.print_summary(detections)
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()