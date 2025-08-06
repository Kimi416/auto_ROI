#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pythonベース画像アノテーションツール
- OpenCVを使用した病変領域の選択
- YOLO形式での保存
- 切り出し画像の即時確認
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import os

class LesionAnnotator:
    def __init__(self, images_dir, output_dir="annotations_output"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "cropped").mkdir(exist_ok=True)
        (self.output_dir / "visualized").mkdir(exist_ok=True)
        
        # 画像ファイル取得
        self.image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
            self.image_files.extend(list(self.images_dir.rglob(ext)))
        self.image_files.sort()
        
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        self.original_image = None
        
        # アノテーション情報
        self.annotations = []
        self.current_annotation = []
        self.drawing = False
        
        # 病変タイプ
        self.disease_types = [
            'Melasma (肝斑)',
            'Solar_lentigo (日光性色素斑)',
            'Nevus (母斑)',
            'ADM (後天性真皮メラノサイトーシス)',
            'Ephelis (雀卵斑)',
            'Seborrheic_keratosis (脂漏性角化症)',
            'Basal_cell_carcinoma (基底細胞癌)',
            'Malignant_melanoma (悪性黒色腫)'
        ]
        self.current_disease = 0
        
        # カラーマップ
        self.colors = [
            (0, 0, 255),    # 赤
            (0, 255, 0),    # 緑
            (255, 0, 0),    # 青
            (0, 255, 255),  # 黄
            (255, 0, 255),  # マゼンタ
            (255, 255, 0),  # シアン
            (128, 0, 255),  # 紫
            (255, 128, 0)   # オレンジ
        ]
        
        # 進捗管理
        self.progress_file = self.output_dir / "progress.json"
        self.load_progress()
        
    def load_progress(self):
        """進捗情報を読み込み"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.current_index = progress.get('last_index', 0)
                print(f"前回の続きから開始: {self.current_index + 1}/{len(self.image_files)}")
    
    def save_progress(self):
        """進捗を保存"""
        progress = {
            'last_index': self.current_index,
            'total': len(self.image_files),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベント処理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_annotation = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_annotation.append((x, y))
                # リアルタイム描画
                if len(self.current_annotation) > 1:
                    self.display_image = self.current_image.copy()
                    self.draw_annotations()
                    pts = np.array(self.current_annotation, np.int32)
                    cv2.polylines(self.display_image, [pts], False, 
                                self.colors[self.current_disease], 2)
                    
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                if len(self.current_annotation) > 3:
                    # 閉じたポリゴンにする
                    self.annotations.append({
                        'points': self.current_annotation.copy(),
                        'disease': self.current_disease
                    })
                    self.current_annotation = []
                    self.display_image = self.current_image.copy()
                    self.draw_annotations()
    
    def draw_annotations(self):
        """全アノテーションを描画"""
        for ann in self.annotations:
            pts = np.array(ann['points'], np.int32)
            cv2.fillPoly(self.display_image, [pts], 
                        (*self.colors[ann['disease']], 100))
            cv2.polylines(self.display_image, [pts], True, 
                         self.colors[ann['disease']], 2)
            
            # 病変タイプラベル
            if len(ann['points']) > 0:
                x, y = ann['points'][0]
                label = self.disease_types[ann['disease']].split('(')[0]
                cv2.putText(self.display_image, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           self.colors[ann['disease']], 2)
    
    def save_annotations(self):
        """現在の画像のアノテーションを保存"""
        if not self.annotations:
            return
        
        img_path = self.image_files[self.current_index]
        base_name = img_path.stem
        h, w = self.original_image.shape[:2]
        
        # YOLO形式で保存
        label_file = self.output_dir / "labels" / f"{base_name}.txt"
        yolo_lines = []
        
        for i, ann in enumerate(self.annotations):
            if len(ann['points']) < 3:
                continue
            
            # バウンディングボックス計算
            pts = np.array(ann['points'])
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            
            # YOLO形式（正規化）
            cx = (x_min + x_max) / 2 / w
            cy = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            
            yolo_lines.append(f"{ann['disease']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            # 病変領域切り出し
            margin = 10
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)
            
            cropped = self.original_image[y1:y2, x1:x2]
            disease_name = self.disease_types[ann['disease']].split('(')[0].strip()
            crop_file = self.output_dir / "cropped" / f"{base_name}_{i}_{disease_name}.jpg"
            cv2.imwrite(str(crop_file), cropped)
        
        # ラベルファイル保存
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # 可視化画像保存
        vis_file = self.output_dir / "visualized" / f"{base_name}_annotated.jpg"
        cv2.imwrite(str(vis_file), self.display_image)
        
        print(f"保存完了: {base_name} ({len(self.annotations)}個の病変)")
    
    def load_image(self):
        """現在のインデックスの画像を読み込み"""
        if 0 <= self.current_index < len(self.image_files):
            img_path = self.image_files[self.current_index]
            self.original_image = cv2.imread(str(img_path))
            
            # リサイズ（表示用）
            h, w = self.original_image.shape[:2]
            max_size = 800
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                self.current_image = cv2.resize(self.original_image, (new_w, new_h))
            else:
                self.current_image = self.original_image.copy()
            
            self.display_image = self.current_image.copy()
            self.annotations = []
            
            # 既存のアノテーション確認
            base_name = img_path.stem
            label_file = self.output_dir / "labels" / f"{base_name}.txt"
            if label_file.exists():
                print(f"既存のアノテーションあり: {base_name}")
    
    def run(self):
        """メインループ"""
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        self.load_image()
        
        print("\n=== 操作方法 ===")
        print("マウス左ドラッグ: 病変領域を描画")
        print("1-8: 病変タイプ選択")
        print("n/→: 次の画像")
        print("p/←: 前の画像")
        print("s: 保存")
        print("c: クリア")
        print("z: 最後のアノテーションを削除")
        print("q/ESC: 終了")
        print("================\n")
        
        while True:
            # 情報表示
            info_img = self.display_image.copy()
            
            # ステータス表示
            status = f"Image: {self.current_index + 1}/{len(self.image_files)}"
            disease_name = self.disease_types[self.current_disease]
            status += f" | Disease: {disease_name}"
            status += f" | Annotations: {len(self.annotations)}"
            
            cv2.putText(info_img, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_img, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            cv2.imshow('Annotation Tool', info_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # キーボード操作
            if key == ord('q') or key == 27:  # ESC
                self.save_progress()
                break
                
            elif key == ord('n') or key == 83:  # 右矢印
                self.save_annotations()
                self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
                self.load_image()
                self.save_progress()
                
            elif key == ord('p') or key == 81:  # 左矢印
                self.save_annotations()
                self.current_index = max(self.current_index - 1, 0)
                self.load_image()
                self.save_progress()
                
            elif key == ord('s'):
                self.save_annotations()
                print("アノテーション保存完了")
                
            elif key == ord('c'):
                self.annotations = []
                self.display_image = self.current_image.copy()
                print("アノテーションをクリア")
                
            elif key == ord('z'):
                if self.annotations:
                    self.annotations.pop()
                    self.display_image = self.current_image.copy()
                    self.draw_annotations()
                    print("最後のアノテーションを削除")
                    
            elif ord('1') <= key <= ord('8'):
                self.current_disease = key - ord('1')
                print(f"病変タイプ: {self.disease_types[self.current_disease]}")
        
        cv2.destroyAllWindows()
        print("\nアノテーション作業終了")
        print(f"出力先: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Python画像アノテーションツール')
    parser.add_argument('images_dir', help='画像ディレクトリ')
    parser.add_argument('-o', '--output', default='annotations_output', help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    annotator = LesionAnnotator(args.images_dir, args.output)
    print(f"画像数: {len(annotator.image_files)}")
    
    if len(annotator.image_files) == 0:
        print("画像が見つかりません")
        return
    
    annotator.run()

if __name__ == '__main__':
    main()