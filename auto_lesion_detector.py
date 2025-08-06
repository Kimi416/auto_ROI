#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO自動病変検出・切り抜きツール
学習済みYOLOモデルを使用して病変を自動検出し、透過PNGで切り抜き
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import json
import argparse
from pathlib import Path

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AutoLesionDetector:
    def __init__(self, model_path='best.pt'):
        """
        Args:
            model_path: 学習済みYOLOモデルのパス
        """
        print("🤖 YOLO自動病変検出システム起動中...")
        
        # モデル読み込み
        if not os.path.exists(model_path):
            # 代替パスを探す
            alt_paths = [
                'runs/detect/train/weights/best.pt',
                'runs/detect/train2/weights/best.pt',
                'runs/detect/train3/weights/best.pt',
                'yolo_training/runs/detect/train/weights/best.pt'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✅ モデル読み込み完了: {model_path}")
        
        # クラス名
        self.class_names = [
            'ADM', 'Ephelis', 'Melasma', 'Solar lentigo',
            'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis',
            'Malignant melanoma'
        ]
        
    def detect_lesions(self, image_path, conf_threshold=0.5):
        """
        画像から病変を検出
        
        Args:
            image_path: 入力画像パス
            conf_threshold: 検出信頼度閾値
            
        Returns:
            results: YOLO検出結果
            image: 元画像（numpy array）
        """
        print(f"\n🔍 病変検出中: {image_path}")
        
        # 画像読み込み
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO検出実行
        results = self.model(image_path, conf=conf_threshold)
        
        # 検出数表示
        if len(results[0].boxes) > 0:
            print(f"✅ {len(results[0].boxes)}個の病変を検出しました")
        else:
            print("⚠️ 病変が検出されませんでした")
            
        return results[0], image_rgb
        
    def visualize_detections(self, image, result, save_path=None):
        """
        検出結果を可視化
        
        Args:
            image: 元画像
            result: YOLO検出結果
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # バウンディングボックス座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 矩形描画
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # ラベル表示
                label = f'{self.class_names[cls]} {conf:.2f}'
                ax.text(x1, y1-5, label, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                       fontsize=10)
        
        ax.set_title(f'自動病変検出結果 ({len(boxes) if result.boxes is not None else 0}個検出)')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 検出結果保存: {save_path}")
        else:
            plt.show()
            
    def extract_lesions(self, image_path, output_dir='extracted_lesions', conf_threshold=0.5):
        """
        検出した病変を個別に切り抜いて透過PNGで保存
        
        Args:
            image_path: 入力画像パス
            output_dir: 出力ディレクトリ
            conf_threshold: 検出信頼度閾値
            
        Returns:
            extracted_count: 切り抜いた病変数
        """
        # 検出実行
        result, image_rgb = self.detect_lesions(image_path, conf_threshold)
        
        if result.boxes is None or len(result.boxes) == 0:
            return 0
            
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # PIL画像に変換
        pil_image = Image.fromarray(image_rgb)
        base_name = Path(image_path).stem
        
        extracted_info = []
        
        for i, box in enumerate(result.boxes):
            # バウンディングボックス座標
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # マージンを追加（10%）
            h, w = image_rgb.shape[:2]
            margin = 0.1
            dx = int((x2 - x1) * margin)
            dy = int((y2 - y1) * margin)
            
            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)
            
            # 病変部分を切り抜き
            lesion_crop = pil_image.crop((x1, y1, x2, y2))
            
            # 透過PNG作成（楕円形マスク）
            mask = Image.new('L', lesion_crop.size, 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            
            # 楕円形マスク作成
            draw.ellipse([0, 0, lesion_crop.size[0], lesion_crop.size[1]], fill=255)
            
            # RGBA画像作成
            rgba_image = lesion_crop.convert('RGBA')
            rgba_image.putalpha(mask)
            
            # 保存
            output_path = os.path.join(
                output_dir, 
                f"{base_name}_lesion_{i+1}_{self.class_names[cls]}.png"
            )
            rgba_image.save(output_path)
            
            # 情報記録
            extracted_info.append({
                'index': i + 1,
                'class': self.class_names[cls],
                'confidence': float(conf),
                'bbox': {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                },
                'output_file': os.path.basename(output_path)
            })
            
        # 抽出情報を保存
        info_path = os.path.join(output_dir, f"{base_name}_extraction_info.json")
        with open(info_path, 'w') as f:
            json.dump({
                'source_image': image_path,
                'total_lesions': len(extracted_info),
                'conf_threshold': conf_threshold,
                'lesions': extracted_info
            }, f, indent=2)
            
        print(f"✅ {len(extracted_info)}個の病変を切り抜きました")
        print(f"📁 保存先: {output_dir}")
        
        return len(extracted_info)
        
    def process_directory(self, input_dir, output_dir='extracted_lesions', 
                         conf_threshold=0.5, visualize=False):
        """
        ディレクトリ内の全画像を処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            conf_threshold: 検出信頼度閾値
            visualize: 検出結果を可視化するか
        """
        input_path = Path(input_dir)
        
        # 画像ファイル収集
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
            
        if not image_files:
            print(f"⚠️ {input_dir} に画像ファイルが見つかりません")
            return
            
        print(f"\n📂 {len(image_files)}枚の画像を処理します")
        
        total_extracted = 0
        results_summary = []
        
        for img_path in image_files:
            print(f"\n処理中: {img_path.name}")
            
            try:
                # 病変抽出
                count = self.extract_lesions(
                    str(img_path), 
                    output_dir=output_dir,
                    conf_threshold=conf_threshold
                )
                total_extracted += count
                
                # 可視化
                if visualize:
                    result, image_rgb = self.detect_lesions(str(img_path), conf_threshold)
                    viz_path = os.path.join(output_dir, f"{img_path.stem}_detection.jpg")
                    self.visualize_detections(image_rgb, result, save_path=viz_path)
                    
                results_summary.append({
                    'image': img_path.name,
                    'lesions_detected': count,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"❌ エラー: {e}")
                results_summary.append({
                    'image': img_path.name,
                    'lesions_detected': 0,
                    'status': 'error',
                    'error': str(e)
                })
                
        # サマリー保存
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'total_lesions_extracted': total_extracted,
                'conf_threshold': conf_threshold,
                'results': results_summary
            }, f, indent=2)
            
        print(f"\n🎉 処理完了!")
        print(f"📊 結果: {len(image_files)}枚の画像から{total_extracted}個の病変を抽出")
        print(f"📁 保存先: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='YOLO自動病変検出・切り抜きツール')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-m', '--model', default='best.pt', help='YOLOモデルのパス')
    parser.add_argument('-o', '--output', default='extracted_lesions', help='出力ディレクトリ')
    parser.add_argument('-c', '--conf', type=float, default=0.5, help='検出信頼度閾値 (0-1)')
    parser.add_argument('-v', '--visualize', action='store_true', help='検出結果を可視化')
    
    args = parser.parse_args()
    
    # 検出器初期化
    detector = AutoLesionDetector(model_path=args.model)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイル処理
        count = detector.extract_lesions(
            str(input_path),
            output_dir=args.output,
            conf_threshold=args.conf
        )
        
        if args.visualize:
            result, image_rgb = detector.detect_lesions(str(input_path), args.conf)
            detector.visualize_detections(image_rgb, result)
            
    elif input_path.is_dir():
        # ディレクトリ処理
        detector.process_directory(
            str(input_path),
            output_dir=args.output,
            conf_threshold=args.conf,
            visualize=args.visualize
        )
    else:
        print(f"❌ エラー: {input_path} が見つかりません")

if __name__ == "__main__":
    main()