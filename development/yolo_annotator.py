#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO学習用の皮膚病変アノテーションツール
バッチ処理でマスク済み画像から病変部分をマーキング
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

class YOLOAnnotationTool:
    """YOLO用病変アノテーションツール"""
    
    def __init__(self, save_interval=10):
        self.save_interval = save_interval  # 自動保存間隔
        self.all_annotations = []  # 全アノテーションデータ
        self.current_image_index = 0
        self.total_images = 0
        
        # カテゴリマッピング
        self.category_mapping = {
            'ADM': 0,
            'Ephelis': 1,
            'Melasma': 2,
            'Solar lentigo': 3,
            'Nevus': 4,
            'Basal cell carcinoma': 5,
            'Seborrheic keratosis': 6,
            'Malignant melanoma': 7
        }
        
        # 作業用変数
        self.image = None
        self.original_image = None
        self.current_bboxes = []
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        self.mode = 'rectangle'  # 'rectangle' or 'freehand'
        self.current_points = []  # フリーハンド用のポイント
        
        # ウィンドウ設定
        self.window_name = "YOLO Skin Lesion Annotator"
        self.max_display_width = 1200
        self.max_display_height = 800
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントのコールバック"""
        if self.mode == 'rectangle':
            self.rectangle_callback(event, x, y, flags, param)
        elif self.mode == 'freehand':
            self.freehand_callback(event, x, y, flags, param)
    
    def rectangle_callback(self, event, x, y, flags, param):
        """矩形選択のコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_bbox = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_bbox[2] = x
                self.current_bbox[3] = y
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_bbox:
                # 最小サイズチェック
                width = abs(self.current_bbox[2] - self.current_bbox[0])
                height = abs(self.current_bbox[3] - self.current_bbox[1])
                
                if width > 5 and height > 5:  # 最小サイズ5x5
                    # 座標を正規化
                    x1 = min(self.current_bbox[0], self.current_bbox[2])
                    y1 = min(self.current_bbox[1], self.current_bbox[3])
                    x2 = max(self.current_bbox[0], self.current_bbox[2])
                    y2 = max(self.current_bbox[1], self.current_bbox[3])
                    
                    self.current_bboxes.append({
                        'type': 'rectangle',
                        'bbox': [x1, y1, x2, y2]
                    })
                    print(f"病変 #{len(self.current_bboxes)} (矩形) を追加しました")
                
                self.current_bbox = None
                self.update_display()
    
    def freehand_callback(self, event, x, y, flags, param):
        """フリーハンド選択のコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_points = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_points.append((x, y))
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.current_points) > 2:
                # フリーハンドの輪郭からバウンディングボックスを計算
                points = np.array(self.current_points, np.int32)
                x, y, w, h = cv2.boundingRect(points)
                
                self.current_bboxes.append({
                    'type': 'freehand',
                    'points': self.current_points,
                    'bbox': [x, y, x + w, y + h]
                })
                print(f"病変 #{len(self.current_bboxes)} (フリーハンド) を追加しました")
            
            self.current_points = []
            self.update_display()
    
    def resize_image_for_display(self, image):
        """表示用に画像をリサイズ"""
        h, w = image.shape[:2]
        
        # アスペクト比を保持してリサイズ
        if w > self.max_display_width or h > self.max_display_height:
            scale_w = self.max_display_width / w
            scale_h = self.max_display_height / h
            scale = min(scale_w, scale_h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            return cv2.resize(image, (new_w, new_h)), scale
        
        return image.copy(), 1.0
    
    def update_display(self):
        """表示を更新"""
        display_image = self.image.copy()
        
        # 既存のバウンディングボックスを描画
        for i, bbox_info in enumerate(self.current_bboxes):
            bbox = bbox_info['bbox']
            x1, y1, x2, y2 = bbox
            
            # タイプに応じて色を変更
            color = (0, 255, 255) if bbox_info['type'] == 'rectangle' else (255, 0, 255)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # フリーハンドの場合は輪郭も描画
            if bbox_info['type'] == 'freehand' and 'points' in bbox_info:
                points = np.array(bbox_info['points'], np.int32)
                cv2.polylines(display_image, [points], True, color, 1)
            
            # 番号を表示
            cv2.putText(display_image, f"#{i+1}({bbox_info['type'][0]})", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 現在描画中のボックス（矩形モード）
        if self.mode == 'rectangle' and self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 現在描画中のフリーハンド
        if self.mode == 'freehand' and len(self.current_points) > 1:
            points = np.array(self.current_points, np.int32)
            cv2.polylines(display_image, [points], False, (0, 255, 0), 2)
        
        # 情報テキストを表示
        info_text = [
            f"Image: {self.current_image_index + 1}/{self.total_images}",
            f"Mode: {self.mode.upper()}",
            f"Lesions: {len(self.current_bboxes)}",
            "",
            "Controls:",
            "Mouse: Draw annotation",
            "R: Rectangle mode",
            "F: Freehand mode", 
            "ENTER: Save and next",
            "SPACE: Skip (no lesions)",
            "D: Delete last box",
            "C: Clear all boxes",
            "ESC: Save and exit"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 20
            cv2.putText(display_image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display_image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow(self.window_name, display_image)
    
    def convert_to_yolo_format(self, bbox, img_width, img_height, scale):
        """バウンディングボックスをYOLO形式に変換"""
        # 表示座標を元画像座標に変換
        x1, y1, x2, y2 = bbox
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # YOLO形式に変換 (正規化された中心座標と幅・高さ)
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return x_center, y_center, width, height
    
    def annotate_image(self, image_path, category):
        """単一画像のアノテーション"""
        # 画像を読み込み
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # 表示用にリサイズ
        self.image, scale = self.resize_image_for_display(self.original_image)
        self.current_bboxes = []
        
        # ウィンドウ設定
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\n📍 アノテーション中: {image_path.name}")
        print(f"カテゴリ: {category}")
        print("病変部分をマウスドラッグで囲んでください")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter - 保存して次へ
                break
            elif key == 32:  # Space - スキップ
                self.current_bboxes = []
                break
            elif key == ord('r'):  # Rectangle mode
                self.mode = 'rectangle'
                print("矩形選択モードに切り替えました")
                self.update_display()
            elif key == ord('f'):  # Freehand mode
                self.mode = 'freehand'
                print("フリーハンド選択モードに切り替えました")
                self.update_display()
            elif key == ord('d'):  # 最後のボックスを削除
                if self.current_bboxes:
                    deleted = self.current_bboxes.pop()
                    print(f"病変 #{len(self.current_bboxes)+1} を削除しました")
                    self.update_display()
            elif key == ord('c'):  # 全クリア
                self.current_bboxes = []
                print("全ての病変をクリアしました")
                self.update_display()
            elif key == 27:  # ESC - 終了
                return False
        
        # アノテーションデータを保存
        if self.current_bboxes:
            orig_h, orig_w = self.original_image.shape[:2]
            class_id = self.category_mapping.get(category, 0)
            
            yolo_annotations = []
            for bbox_info in self.current_bboxes:
                bbox = bbox_info['bbox']
                x_center, y_center, width, height = self.convert_to_yolo_format(
                    bbox, orig_w, orig_h, scale
                )
                yolo_annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'type': bbox_info['type']  # 矩形かフリーハンドかを記録
                })
            
            annotation_data = {
                'image_path': str(image_path),
                'category': category,
                'class_id': class_id,
                'lesion_count': len(self.current_bboxes),
                'annotations': yolo_annotations,
                'timestamp': datetime.now().isoformat()
            }
            
            self.all_annotations.append(annotation_data)
            print(f"✅ {len(self.current_bboxes)}個の病変をアノテーションしました")
        else:
            print("⏭️  病変なしでスキップしました")
        
        return True
    
    def save_annotations(self, output_file):
        """アノテーションデータを保存"""
        output_path = Path(output_file)
        
        # JSON形式で保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 アノテーションデータを保存しました: {output_path}")
        
        # 統計情報を表示
        total_images = len(self.all_annotations)
        total_lesions = sum(item['lesion_count'] for item in self.all_annotations)
        
        print(f"\n📊 アノテーション統計:")
        print(f"  総画像数: {total_images}")
        print(f"  総病変数: {total_lesions}")
        
        # カテゴリ別統計
        category_stats = {}
        for item in self.all_annotations:
            cat = item['category']
            category_stats[cat] = category_stats.get(cat, 0) + item['lesion_count']
        
        print(f"\n📈 カテゴリ別病変数:")
        for cat, count in sorted(category_stats.items()):
            print(f"  {cat}: {count}")
    
    def load_existing_annotations(self, output_file):
        """既存のアノテーションファイルを読み込み"""
        try:
            if Path(output_file).exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    self.all_annotations = json.load(f)
                print(f"✅ 既存のアノテーション {len(self.all_annotations)}件を読み込みました")
                return set(item['image_path'] for item in self.all_annotations)
            return set()
        except Exception as e:
            print(f"⚠️ アノテーション読み込みエラー: {e}")
            return set()

    def process_directory(self, input_dir, output_file="yolo_annotations.json"):
        """ディレクトリ内の画像を順次アノテーション"""
        input_path = Path(input_dir)
        
        # 既存のアノテーションを読み込み
        processed_images = self.load_existing_annotations(output_file)
        
        # 画像ファイルを収集（カテゴリ別）
        category_images = {}
        
        for category_dir in input_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                if category in self.category_mapping:
                    image_files = []
                    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
                        image_files.extend(category_dir.glob(ext))
                    
                    # 未処理の画像のみをフィルタ
                    unprocessed_files = [f for f in image_files if str(f) not in processed_images]
                    
                    if unprocessed_files:
                        category_images[category] = sorted(unprocessed_files)
        
        # 総画像数を計算
        self.total_images = sum(len(images) for images in category_images.values())
        
        print(f"🎯 YOLO皮膚病変アノテーションツール")
        print(f"📁 入力ディレクトリ: {input_dir}")
        print(f"📊 残り画像数: {self.total_images}")
        print(f"✅ 処理済み: {len(processed_images)}枚")
        print(f"\nカテゴリ別残り画像数:")
        for cat, images in category_images.items():
            print(f"  {cat}: {len(images)}枚")
        
        print(f"\n🚀 アノテーション開始!")
        
        # カテゴリごとに処理
        self.current_image_index = 0
        
        try:
            for category, image_files in category_images.items():
                print(f"\n{'='*50}")
                print(f"📂 カテゴリ: {category} ({len(image_files)}枚)")
                print(f"{'='*50}")
                
                for image_path in image_files:
                    if not self.annotate_image(image_path, category):
                        # ESCで終了
                        break
                    
                    self.current_image_index += 1
                    
                    # 定期的に自動保存
                    if self.current_image_index % self.save_interval == 0:
                        self.save_annotations(output_file)
                        print(f"🔄 進捗: {self.current_image_index}/{self.total_images} 自動保存完了")
                else:
                    continue
                break  # ESCで終了した場合
        
        except KeyboardInterrupt:
            print("\n⚠️ Ctrl+Cで中断されました")
        
        finally:
            # 最終保存
            cv2.destroyAllWindows()
            if self.all_annotations:
                self.save_annotations(output_file)
            
            print(f"\n🏁 アノテーション作業完了!")
            print(f"📝 処理済み: {len(self.all_annotations)}/{self.total_images}")

def main():
    parser = argparse.ArgumentParser(description='YOLO皮膚病変アノテーションツール')
    parser.add_argument('input_dir', help='マスク済み画像ディレクトリのパス')
    parser.add_argument('-o', '--output', default='yolo_annotations.json',
                        help='出力アノテーションファイル')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='自動保存間隔（画像数）')
    
    args = parser.parse_args()
    
    # アノテーターを初期化
    annotator = YOLOAnnotationTool(save_interval=args.save_interval)
    
    # バッチ処理を開始
    annotator.process_directory(args.input_dir, args.output)

if __name__ == "__main__":
    main()