#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安定版画像アノテーションツール
- メモリリーク対策
- エラー回復機能
- 自動再起動機能
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import os
import gc
import traceback
import sys

class StableAnnotator:
    def __init__(self, images_dir, output_dir="annotations_output"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "cropped").mkdir(exist_ok=True)
        
        # 画像ファイル取得
        self.image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
            self.image_files.extend(list(self.images_dir.rglob(ext)))
        self.image_files.sort()
        
        self.current_index = 0
        self.current_image = None
        self.original_image = None
        self.current_filename = ""  # 現在のファイル名
        
        # アノテーション
        self.annotations = []
        self.current_points = []
        self.drawing = False
        
        # 病変タイプ（8種類）
        self.disease_types = [
            'Melasma',              # 1. 肝斑
            'Solar_lentigo',        # 2. 日光性色素斑
            'Nevus',                # 3. 母斑
            'ADM',                  # 4. 後天性真皮メラノサイトーシス
            'Ephelis',              # 5. 雀卵斑
            'Seborrheic_keratosis', # 6. 脂漏性角化症
            'Basal_cell_carcinoma', # 7. 基底細胞癌（BCC）
            'Malignant_melanoma'    # 8. 悪性黒色腫
        ]
        self.current_disease = 0
        self.colors = [
            (0,0,255),    # 赤 - Melasma
            (0,255,0),    # 緑 - Solar_lentigo
            (255,0,0),    # 青 - Nevus
            (0,255,255),  # 黄 - ADM
            (255,0,255),  # マゼンタ - Ephelis
            (255,128,0),  # オレンジ - Seborrheic_keratosis
            (128,0,255),  # 紫 - Basal_cell_carcinoma
            (0,128,255)   # 水色 - Malignant_melanoma
        ]
        
        # 進捗管理
        self.progress_file = self.output_dir / "progress.json"
        self.load_progress()
        
        # 自動保存カウンター
        self.auto_save_interval = 5  # 5枚ごとに自動保存
        self.images_processed = 0
        
    def load_progress(self):
        """進捗を読み込み"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.current_index = progress.get('last_index', 0)
                    print(f"📌 続きから開始: {self.current_index + 1}/{len(self.image_files)}")
        except Exception as e:
            print(f"進捗読み込みエラー: {e}")
            self.current_index = 0
    
    def save_progress(self):
        """進捗を保存"""
        try:
            progress = {
                'last_index': self.current_index,
                'total': len(self.image_files),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"進捗保存エラー: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベント処理（簡略化）"""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.current_points = [(x, y)]
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.current_points.append((x, y))
                # 描画中の線を表示
                temp = self.current_image.copy()
                if len(self.current_points) > 1:
                    pts = np.array(self.current_points, np.int32)
                    cv2.polylines(temp, [pts], False, (0, 255, 255), 3)
                cv2.imshow('Annotation', temp)
                
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing and len(self.current_points) > 3:
                    self.annotations.append({
                        'points': self.current_points.copy(),
                        'disease': self.current_disease
                    })
                    print(f"✅ 病変追加 (合計: {len(self.annotations)})")
                self.drawing = False
                self.current_points = []
                self.redraw()
        except Exception as e:
            print(f"マウスエラー: {e}")
            self.drawing = False
    
    def redraw(self):
        """画像を再描画"""
        try:
            if self.current_image is None:
                return
            
            temp = self.current_image.copy()
            
            # アノテーションを描画
            for i, ann in enumerate(self.annotations):
                pts = np.array(ann['points'], np.int32)
                color = self.colors[ann['disease'] % len(self.colors)]
                cv2.polylines(temp, [pts], True, color, 2)
                cv2.fillPoly(temp, [pts], (*color, 50))
                
                # ラベル
                if len(ann['points']) > 0:
                    x, y = ann['points'][0]
                    label = f"{i+1}"
                    cv2.putText(temp, label, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # ファイル名を画像上部に表示
            h, w = temp.shape[:2]
            cv2.rectangle(temp, (0, 0), (w, 30), (50, 50, 50), -1)
            file_info = f"{self.current_filename} [{self.current_index+1}/{len(self.image_files)}]"
            cv2.putText(temp, file_info, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Annotation', temp)
        except Exception as e:
            print(f"描画エラー: {e}")
    
    def load_image(self):
        """現在の画像を読み込み"""
        try:
            # メモリクリア
            if self.original_image is not None:
                del self.original_image
            if self.current_image is not None:
                del self.current_image
            gc.collect()
            
            if 0 <= self.current_index < len(self.image_files):
                img_path = self.image_files[self.current_index]
                self.current_filename = img_path.name  # ファイル名を保存
                print(f"\n📷 画像: {img_path.name}")
                
                self.original_image = cv2.imread(str(img_path))
                if self.original_image is None:
                    print(f"⚠️ 読み込み失敗: {img_path}")
                    return False
                
                # リサイズ（メモリ節約）
                h, w = self.original_image.shape[:2]
                max_size = 500  # さらに小さく
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    self.current_image = cv2.resize(self.original_image, (new_w, new_h))
                else:
                    self.current_image = self.original_image.copy()
                
                self.annotations = []
                return True
        except Exception as e:
            print(f"❌ 画像読み込みエラー: {e}")
            return False
    
    def save_current(self):
        """現在のアノテーションを保存"""
        try:
            if not self.annotations or self.original_image is None:
                return
            
            img_path = self.image_files[self.current_index]
            base_name = img_path.stem
            h, w = self.original_image.shape[:2]
            
            # YOLOラベル保存
            label_file = self.output_dir / "labels" / f"{base_name}.txt"
            yolo_lines = []
            
            for i, ann in enumerate(self.annotations):
                pts = np.array(ann['points'])
                
                # スケール調整（リサイズした座標を元のサイズに）
                scale_x = w / self.current_image.shape[1]
                scale_y = h / self.current_image.shape[0]
                pts[:, 0] = pts[:, 0] * scale_x
                pts[:, 1] = pts[:, 1] * scale_y
                
                # バウンディングボックス
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                
                cx = (x_min + x_max) / 2 / w
                cy = (y_min + y_max) / 2 / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h
                
                yolo_lines.append(f"{ann['disease']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                # 病変切り出し（オプション）
                if i < 3:  # 最初の3つだけ保存（メモリ節約）
                    margin = 10
                    x1 = max(0, int(x_min - margin))
                    y1 = max(0, int(y_min - margin))
                    x2 = min(w, int(x_max + margin))
                    y2 = min(h, int(y_max + margin))
                    
                    cropped = self.original_image[y1:y2, x1:x2]
                    crop_file = self.output_dir / "cropped" / f"{base_name}_{i}.jpg"
                    cv2.imwrite(str(crop_file), cropped)
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            print(f"💾 保存: {base_name} ({len(self.annotations)}個)")
            
        except Exception as e:
            print(f"保存エラー: {e}")
    
    def run(self):
        """メインループ"""
        cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotation', 800, 600)
        cv2.setMouseCallback('Annotation', self.mouse_callback)
        
        if not self.load_image():
            print("最初の画像読み込み失敗")
            return
        
        self.redraw()
        
        print("\n=== 簡易操作 ===")
        print("マウス: 病変を描画")
        print("Space/n: 次へ")
        print("p: 前へ")
        print("1-8: 病変タイプ選択")
        print("  1:肝斑 2:日光性色素斑 3:母斑 4:ADM")
        print("  5:雀卵斑 6:脂漏性角化症 7:BCC 8:悪性黒色腫")
        print("c: クリア")
        print("q: 終了")
        print("===============\n")
        
        while True:
            try:
                # ステータス表示
                status = f"[{self.current_index+1}/{len(self.image_files)}] "
                status += f"Type: {self.disease_types[self.current_disease]} "
                status += f"Lesions: {len(self.annotations)}"
                
                # ウィンドウタイトルに表示
                cv2.setWindowTitle('Annotation', status)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    self.save_current()
                    self.save_progress()
                    break
                    
                elif key == ord(' ') or key == ord('n'):  # 次へ
                    self.save_current()
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                        self.load_image()
                        self.redraw()
                        self.save_progress()
                        
                        # 自動保存
                        self.images_processed += 1
                        if self.images_processed % self.auto_save_interval == 0:
                            print(f"🔄 自動保存 ({self.images_processed}枚処理)")
                            gc.collect()  # メモリクリーンアップ
                    
                elif key == ord('p'):  # 前へ
                    self.save_current()
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.load_image()
                        self.redraw()
                        self.save_progress()
                
                elif key == ord('c'):  # クリア
                    self.annotations = []
                    self.redraw()
                    print("🗑️ クリア")
                
                elif ord('1') <= key <= ord('8'):  # 病変タイプ
                    self.current_disease = key - ord('1')
                    type_name = self.disease_types[self.current_disease]
                    print(f"Type: {type_name} ({key - ord('0')})")
                    
            except KeyboardInterrupt:
                print("\n中断")
                self.save_current()
                self.save_progress()
                break
                
            except Exception as e:
                print(f"エラー: {e}")
                traceback.print_exc()
                # エラー時も継続
                
        cv2.destroyAllWindows()
        print(f"\n✅ 終了 - {self.images_processed}枚処理")

def main():
    parser = argparse.ArgumentParser(description='安定版アノテーションツール')
    parser.add_argument('images_dir', help='画像ディレクトリ')
    parser.add_argument('-o', '--output', default='annotations_output', help='出力先')
    
    args = parser.parse_args()
    
    # 自動再起動機能
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            annotator = StableAnnotator(args.images_dir, args.output)
            print(f"画像数: {len(annotator.image_files)}")
            
            if len(annotator.image_files) == 0:
                print("画像が見つかりません")
                break
            
            annotator.run()
            break  # 正常終了
            
        except Exception as e:
            retry_count += 1
            print(f"\n⚠️ エラー発生 (試行 {retry_count}/{max_retries}): {e}")
            traceback.print_exc()
            
            if retry_count < max_retries:
                print("5秒後に再起動...")
                import time
                time.sleep(5)
            else:
                print("❌ 最大試行回数に達しました")
                break

if __name__ == '__main__':
    main()