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
        self.temp_display = None  # 描画中の一時表示用
        
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
                # リアルタイム描画（滑らかな線）
                if len(self.current_annotation) > 1:
                    # 既存のアノテーションを含む画像をベースに
                    temp_image = self.current_image.copy()
                    self.draw_annotations_on_image(temp_image)
                    
                    # 現在描画中の線を表示
                    pts = np.array(self.current_annotation, np.int32)
                    # 描画中の線を太く明るい色で
                    cv2.polylines(temp_image, [pts], False, 
                                (0, 255, 255), 4)  # 黄色で太く
                    # 始点を円で強調
                    cv2.circle(temp_image, self.current_annotation[0], 5, (0, 255, 0), -1)
                    
                    self.temp_display = temp_image
                    
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
                    print(f"病変追加: {self.disease_types[self.current_disease].split('(')[0]} (合計: {len(self.annotations)}個)")
    
    def draw_annotations(self):
        """全アノテーションを描画（display_imageに）"""
        self.draw_annotations_on_image(self.display_image)
    
    def draw_annotations_on_image(self, image):
        """指定された画像にアノテーションを描画"""
        for i, ann in enumerate(self.annotations):
            pts = np.array(ann['points'], np.int32)
            # 半透明の塗りつぶし
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], self.colors[ann['disease']])
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            # 輪郭線を太く
            cv2.polylines(image, [pts], True, 
                         self.colors[ann['disease']], 3)
            
            # 病変番号とタイプラベル
            if len(ann['points']) > 0:
                x, y = ann['points'][0]
                label = f"{i+1}. {self.disease_types[ann['disease']].split('(')[0]}"
                # 背景付きテキスト
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x-2, y-h-5), (x+w+2, y), (255, 255, 255), -1)
                cv2.putText(image, label, (x, y-5),
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
        try:
            if 0 <= self.current_index < len(self.image_files):
                img_path = self.image_files[self.current_index]
                self.original_image = cv2.imread(str(img_path))
                
                if self.original_image is None:
                    print(f"⚠️ 画像読み込み失敗: {img_path}")
                    return False
                
                # メモリ解放
                if self.current_image is not None:
                    del self.current_image
                if self.display_image is not None:
                    del self.display_image
                if self.temp_display is not None:
                    del self.temp_display
                    self.temp_display = None
                
                # リサイズ（表示用）- より小さいサイズに
                h, w = self.original_image.shape[:2]
                max_size = 600  # 800から600に縮小
                if w > max_size or h > max_size:
                    scale = max_size / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    self.current_image = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    self.current_image = self.original_image.copy()
                
                self.display_image = self.current_image.copy()
                self.annotations = []
                
                # 既存のアノテーション確認
                base_name = img_path.stem
                label_file = self.output_dir / "labels" / f"{base_name}.txt"
                if label_file.exists():
                    print(f"既存のアノテーションあり: {base_name}")
                
                return True
        except Exception as e:
            print(f"❌ エラー: {e}")
            return False
    
    def run(self):
        """メインループ"""
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        self.load_image()
        
        print("\n=== 操作方法 ===")
        print("🖱️  マウス左ドラッグ: フリーハンドで病変領域を描画")
        print("⌨️  1-8: 病変タイプ選択")
        print("➡️  n/→/スペース: 次の画像へ（自動保存）")
        print("⬅️  p/←: 前の画像へ")
        print("💾 s: 現在の画像を保存")
        print("🗑️  c: 全クリア | z: 最後の病変を削除")
        print("❌ q/ESC: 終了")
        print("================\n")
        print("💡 ヒント: 複数の病変を次々に描画できます")
        
        while True:
            try:
                # 描画中の場合は一時表示を使用
                if self.drawing and self.temp_display is not None:
                    info_img = self.temp_display.copy()
                else:
                    info_img = self.display_image.copy()
            except Exception as e:
                print(f"⚠️ 表示エラー: {e}")
                self.load_image()
                continue
            
            # ステータス表示（より見やすく）
            h, w = info_img.shape[:2]
            
            # 上部ステータスバー
            cv2.rectangle(info_img, (0, 0), (w, 50), (50, 50, 50), -1)
            
            status_text = f"[{self.current_index + 1}/{len(self.image_files)}] "
            status_text += f"病変: {len(self.annotations)}個 | "
            status_text += f"選択中: {self.disease_types[self.current_disease].split('(')[0]}"
            
            cv2.putText(info_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ショートカットヒント（右上）
            hint = "Space: 次へ | z: 削除"
            cv2.putText(info_img, hint, (w-250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Annotation Tool', info_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # キーボード操作
            if key == ord('q') or key == 27:  # ESC
                self.save_annotations()
                self.save_progress()
                break
                
            elif key == ord('n') or key == 83 or key == ord(' '):  # 右矢印またはスペース
                try:
                    self.save_annotations()
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                        if not self.load_image():
                            print("⚠️ 画像読み込みエラー、次の画像へ")
                            self.current_index += 1
                            self.load_image()
                        self.save_progress()
                        print(f"📸 画像 {self.current_index + 1}/{len(self.image_files)}")
                    else:
                        print("✅ 最後の画像です")
                except Exception as e:
                    print(f"❌ エラー: {e}")
                
            elif key == ord('p') or key == 81:  # 左矢印
                try:
                    self.save_annotations()
                    if self.current_index > 0:
                        self.current_index -= 1
                        if not self.load_image():
                            print("⚠️ 画像読み込みエラー、前の画像へ")
                            self.current_index -= 1
                            self.load_image()
                        self.save_progress()
                        print(f"📸 画像 {self.current_index + 1}/{len(self.image_files)}")
                except Exception as e:
                    print(f"❌ エラー: {e}")
                
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