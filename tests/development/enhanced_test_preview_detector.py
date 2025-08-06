#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_preview.jpg専用の強化病変検出器
複数の手法を組み合わせて検出精度を向上
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class EnhancedTestPreviewDetector:
    def __init__(self):
        self.model_path = "fast_lesion_training/training_runs/fast_lesion_20250806_095404/weights/best.pt"
        self.model = YOLO(self.model_path)
        
        # クラス定義
        self.pad_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        self.class_names_jp = {
            'ACK': '日光角化症',
            'BCC': '基底細胞癌', 
            'MEL': '悪性黒色腫',
            'NEV': '色素性母斑',
            'SCC': '有棘細胞癌',
            'SEK': '脂漏性角化症'
        }
        
        self.colors_bgr = {
            'ACK': (0, 255, 0),      # 緑
            'BCC': (0, 0, 255),      # 赤
            'MEL': (255, 0, 255),    # マゼンタ
            'NEV': (255, 255, 0),    # シアン
            'SCC': (0, 165, 255),    # オレンジ
            'SEK': (128, 0, 128)     # 紫
        }
    
    def preprocess_image_variants(self, image_path):
        """画像の複数のバリエーションを作成"""
        original = cv2.imread(image_path)
        variants = []
        
        # 1. オリジナル
        variants.append(("original", original))
        
        # 2. コントラスト強化
        enhanced = cv2.convertScaleAbs(original, alpha=1.3, beta=10)
        variants.append(("enhanced_contrast", enhanced))
        
        # 3. ヒストグラム平坦化
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        variants.append(("histogram_equalized", equalized))
        
        # 4. シャープ化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(original, -1, kernel)
        variants.append(("sharpened", sharpened))
        
        # 5. ガンマ補正
        gamma = 0.7
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(original, table)
        variants.append(("gamma_corrected", gamma_corrected))
        
        return variants
    
    def detect_on_variant(self, variant_name, image, conf_thresholds):
        """特定の画像バリアントで検出実行"""
        print(f"🔍 {variant_name}で検出実行中...")
        
        # 一時ファイルに保存
        temp_path = f"temp_{variant_name}.jpg"
        cv2.imwrite(temp_path, image)
        
        best_detections = []
        best_conf = None
        
        for conf in conf_thresholds:
            results = self.model(temp_path, conf=conf, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.pad_classes[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detections.append({
                            'class': class_name,
                            'class_jp': self.class_names_jp[class_name],
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'variant': variant_name,
                            'threshold': conf
                        })
            
            if detections:
                best_detections = detections
                best_conf = conf
                print(f"  ✅ {variant_name}: {len(detections)}個の病変を検出（信頼度: {conf}）")
                break
            else:
                print(f"  ❌ {variant_name}: 信頼度{conf}で検出なし")
        
        # 一時ファイル削除
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return best_detections
    
    def multi_scale_detection(self, image_path):
        """マルチスケール検出"""
        print("🎯 マルチスケール検出開始")
        print("=" * 60)
        
        # 超低信頼度での検出
        ultra_low_conf = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        # 複数バリアントで検出
        variants = self.preprocess_image_variants(image_path)
        all_detections = []
        
        for variant_name, image in variants:
            detections = self.detect_on_variant(variant_name, image, ultra_low_conf)
            all_detections.extend(detections)
        
        if not all_detections:
            print("\n❌ すべてのバリアントで病変が検出されませんでした")
            
            # 最後の手段：画像解析による候補領域検出
            print("\n🔬 画像解析による候補領域検出を試行...")
            candidates = self.analyze_suspicious_regions(image_path)
            return candidates
        
        # 検出結果の統合
        print(f"\n📊 検出結果統合: {len(all_detections)}個の候補")
        merged_detections = self.merge_similar_detections(all_detections)
        
        return merged_detections
    
    def analyze_suspicious_regions(self, image_path):
        """画像解析による疑わしい領域の検出"""
        original = cv2.imread(image_path)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        suspicious_regions = []
        
        # 1. 色の変化が大きい領域を検出
        # HSVに変換
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        # 茶色・黒色系の領域を検出（一般的な皮膚病変の色）
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        combined_mask = cv2.bitwise_or(brown_mask, dark_mask)
        
        # 輪郭検出
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"🔍 {len(contours)}個の候補領域を発見")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # 最小面積フィルター
                x, y, w, h = cv2.boundingRect(contour)
                
                # アスペクト比チェック（極端に細長い領域を除外）
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:
                    suspicious_regions.append({
                        'class': 'UNKNOWN',
                        'class_jp': '疑わしい領域',
                        'confidence': 0.5,  # 固定値
                        'bbox': [x, y, x+w, y+h],
                        'variant': 'image_analysis',
                        'threshold': 'analysis',
                        'area': area
                    })
        
        print(f"📍 {len(suspicious_regions)}個の有効な疑わしい領域を特定")
        return suspicious_regions
    
    def merge_similar_detections(self, detections):
        """類似する検出結果をマージ"""
        if not detections:
            return []
        
        # 信頼度でソート
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        for det in detections:
            # 既存の検出と重複チェック
            is_duplicate = False
            for existing in merged:
                if self.calculate_iou(det['bbox'], existing['bbox']) > 0.3:
                    # 重複している場合、より高い信頼度を保持
                    if det['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(det)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged[:10]  # 最大10個まで
    
    def calculate_iou(self, box1, box2):
        """IoU（Intersection over Union）を計算"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 交差領域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_results(self, image_path, detections):
        """検出結果の可視化"""
        original = cv2.imread(image_path)
        annotated = original.copy()
        
        print(f"\n📊 最終検出結果: {len(detections)}個")
        
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            class_jp = det['class_jp']
            conf = det['confidence']
            variant = det['variant']
            threshold = det['threshold']
            
            color = self.colors_bgr.get(class_name, (255, 255, 255))
            
            print(f"  {i}. {class_jp} ({class_name})")
            print(f"     信頼度: {conf:.3f}")
            print(f"     検出方法: {variant} (閾値: {threshold})")
            print(f"     位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"     サイズ: {x2-x1}px × {y2-y1}px")
            print()
            
            # バウンディングボックス描画
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 6)
            
            # ラベル描画
            label = f"{i}. {class_name} {conf:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            
            # ラベル位置調整
            label_y = y1 - 20
            if label_y < 50:
                label_y = y2 + 50
                
            # ラベル背景
            cv2.rectangle(annotated, 
                        (x1, label_y - label_size[1] - 15), 
                        (x1 + label_size[0] + 15, label_y + 10), 
                        color, -1)
            
            # ラベルテキスト
            cv2.putText(annotated, label, (x1 + 8, label_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # 中央に番号
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 30, color, -1)
            cv2.putText(annotated, str(i), (center_x-15, center_y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        
        # 結果保存
        output_path = "enhanced_test_preview_detection_result.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"💾 強化検出結果保存: {output_path}")
        
        # 比較可視化
        self.create_comparison_visualization(original, annotated, detections)
        
        return detections
    
    def create_comparison_visualization(self, original, annotated, detections):
        """比較可視化を作成"""
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # 元画像
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original test_preview.jpg', fontsize=18, fontweight='bold')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # 強化検出結果
        axes[1].imshow(annotated_rgb)
        axes[1].set_title(f'Enhanced Detection Results\n({len(detections)} lesions detected)', 
                         fontsize=18, fontweight='bold')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig("enhanced_test_preview_comparison.png", dpi=300, bbox_inches='tight')
        print("💾 強化比較可視化保存: enhanced_test_preview_comparison.png")
        plt.show()

def main():
    detector = EnhancedTestPreviewDetector()
    
    print("🚀 test_preview.jpg 強化病変検出開始")
    print("=" * 80)
    
    image_path = "test_preview.jpg"
    detections = detector.multi_scale_detection(image_path)
    
    if detections:
        detector.visualize_results(image_path, detections)
        
        print(f"\n🎉 強化検出完了!")
        print(f"📊 検出された病変数: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_jp']} - 信頼度: {det['confidence']:.3f} ({det['variant']})")
    else:
        print("\n❌ 強化検出でも病変は検出されませんでした")

if __name__ == "__main__":
    main()