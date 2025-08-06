#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
体幹部BCC画像のアノテーション支援ツール
マウスクリックでBCC領域を指定してYOLOフォーマットで保存
"""

import cv2
import numpy as np
from pathlib import Path
import json

class BCCAnnotationTool:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.drawing = False
        self.bbox = []
        self.bboxes = []
        self.current_image_path = None
        self.bcc_class_id = 5  # Basal cell carcinomaのクラスID
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスコールバック関数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.bbox[2] = x
                self.bbox[3] = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if abs(self.bbox[2] - self.bbox[0]) > 10 and abs(self.bbox[3] - self.bbox[1]) > 10:
                self.bboxes.append(self.bbox.copy())
                
    def draw_bboxes(self):
        """バウンディングボックスを描画"""
        self.image = self.original_image.copy()
        
        for bbox in self.bboxes:
            cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(self.image, 'BCC', (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 現在描画中のボックス
        if self.drawing and len(self.bbox) == 4:
            cv2.rectangle(self.image, (self.bbox[0], self.bbox[1]), 
                         (self.bbox[2], self.bbox[3]), (0, 0, 255), 2)
    
    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """OpenCVのバウンディングボックスをYOLOフォーマットに変換"""
        x1, y1, x2, y2 = bbox
        
        # 座標を正規化
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        return center_x, center_y, width, height
    
    def save_annotations(self, output_label_path):
        """アノテーションをYOLOフォーマットで保存"""
        if not self.bboxes:
            print("⚠️ アノテーションがありません")
            return
            
        img_height, img_width = self.original_image.shape[:2]
        
        with open(output_label_path, 'w') as f:
            for bbox in self.bboxes:
                center_x, center_y, width, height = self.convert_to_yolo_format(
                    bbox, img_width, img_height)
                f.write(f"{self.bcc_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ {len(self.bboxes)}個のBCCアノテーションを保存: {output_label_path}")
    
    def annotate_image(self, image_path, output_image_dir, output_label_dir):
        """画像をアノテーション"""
        self.current_image_path = Path(image_path)
        self.original_image = cv2.imread(str(image_path))
        
        if self.original_image is None:
            print(f"❌ 画像を読み込めません: {image_path}")
            return False
            
        self.image = self.original_image.copy()
        self.bboxes = []
        
        # ウィンドウ設定
        window_name = f"BCC Annotation - {self.current_image_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n📸 {self.current_image_path.name}")
        print("操作方法:")
        print("- マウスドラッグでBCC領域を選択")
        print("- 's': 保存して次へ")
        print("- 'u': 最後のボックスを削除")
        print("- 'r': 全てリセット")
        print("- 'q': 保存せずにスキップ")
        print("- ESC: 終了")
        
        while True:
            self.draw_bboxes()
            cv2.imshow(window_name, self.image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # 保存
                output_image_path = output_image_dir / f"bcc_{self.current_image_path.name}"
                output_label_path = output_label_dir / f"bcc_{self.current_image_path.stem}.txt"
                
                # 画像を保存
                cv2.imwrite(str(output_image_path), self.original_image)
                
                # ラベルを保存
                self.save_annotations(output_label_path)
                break
                
            elif key == ord('u'):  # 最後のボックス削除
                if self.bboxes:
                    self.bboxes.pop()
                    print("🗑️ 最後のボックスを削除")
                    
            elif key == ord('r'):  # リセット
                self.bboxes = []
                print("🔄 アノテーションをリセット")
                
            elif key == ord('q'):  # スキップ
                print("⏭️ スキップ")
                break
                
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyWindow(window_name)
        return True

def main():
    print("🔬 体幹部BCC画像アノテーションツール")
    print("="*50)
    
    # ディレクトリ設定
    trunk_bcc_dir = Path('trunk_bcc_images')
    output_image_dir = Path('augmented_dataset/train/images')
    output_label_dir = Path('augmented_dataset/train/labels')
    
    # ディレクトリ作成
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    if not trunk_bcc_dir.exists():
        print(f"❌ {trunk_bcc_dir} フォルダが見つかりません")
        print("体幹部BCC画像を配置してください")
        return
    
    # 画像ファイルを取得
    image_files = list(trunk_bcc_dir.glob("*.jpg")) + list(trunk_bcc_dir.glob("*.png"))
    
    if not image_files:
        print("❌ アノテーション対象の画像が見つかりません")
        return
    
    print(f"📁 {len(image_files)}個の画像が見つかりました")
    
    annotator = BCCAnnotationTool()
    annotated_count = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\n進行状況: {i+1}/{len(image_files)}")
        
        if annotator.annotate_image(image_path, output_image_dir, output_label_dir):
            annotated_count += 1
        else:
            break
    
    print(f"\n🎉 アノテーション完了: {annotated_count}個の画像")
    
    if annotated_count > 0:
        print("\n次のステップ:")
        print("1. python incremental_bcc_training.py を実行")
        print("2. 既存モデルに追加学習を実行")

if __name__ == "__main__":
    main()