#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC専用フリーハンドアノテーションツール
マウスでフリーハンド描画してBCC領域を指定
"""

import cv2
import numpy as np
import json
from pathlib import Path
import random

class BCCFreehandAnnotator:
    def __init__(self):
        self.bcc_class_id = 5
        self.current_image = None
        self.current_original = None
        self.current_path = None
        self.mask = None
        self.drawing = False
        self.brush_size = 15
        
        # BCCファイル一覧を読み込み
        with open('bcc_additions.json', 'r', encoding='utf-8') as f:
            self.bcc_files = json.load(f)
        
        # ランダムシャッフルで変化をつける
        random.shuffle(self.bcc_files)
        self.current_index = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右クリックで消去
            cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
    
    def create_bbox_from_mask(self):
        """マスクからバウンディングボックスを作成"""
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # 小さすぎる領域は無視
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
        
        return bboxes
    
    def convert_to_yolo(self, bbox, img_width, img_height):
        """バウンディングボックスをYOLOフォーマットに変換"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return center_x, center_y, width, height
    
    def save_annotation(self, label_path):
        """アノテーションを保存"""
        bboxes = self.create_bbox_from_mask()
        
        if not bboxes:
            print("⚠️ BCCアノテーションがありません")
            return False
        
        img_height, img_width = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                center_x, center_y, width, height = self.convert_to_yolo(bbox, img_width, img_height)
                f.write(f"{self.bcc_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ {len(bboxes)}個のBCCを保存: {Path(label_path).name}")
        return True
    
    def draw_overlay(self):
        """マスクを重ねた画像を作成"""
        overlay = self.current_image.copy()
        
        # マスクを緑色で重ね合わせ
        green_mask = np.zeros_like(overlay)
        green_mask[self.mask > 0] = [0, 255, 0]
        
        # 半透明で重ね合わせ
        overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
        
        # バウンディングボックスも表示
        bboxes = self.create_bbox_from_mask()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, 'BCC', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay
    
    def show_help(self):
        """ヘルプを表示"""
        print("\n" + "="*60)
        print("🖱️  BCC フリーハンドアノテーション操作方法")
        print("="*60)
        print("左クリック+ドラッグ : BCC領域を描画")
        print("右クリック+ドラッグ : 描画を消去")
        print("マウスホイー↑/↓   : ブラシサイズ変更")
        print("")
        print("キーボード操作:")
        print("  's' : 保存して次へ")
        print("  'q' : スキップして次へ")  
        print("  'r' : 描画をリセット")
        print("  'h' : ヘルプ表示")
        print("  '+'/'-' : ブラシサイズ変更")
        print("  ESC : 終了")
        print("="*60)
    
    def annotate_bcc_files(self):
        """BCCファイルのアノテーション実行"""
        print(f"🔬 BCC フリーハンドアノテーション開始")
        print(f"📁 対象ファイル数: {len(self.bcc_files)}枚")
        
        self.show_help()
        
        while self.current_index < len(self.bcc_files):
            file_info = self.bcc_files[self.current_index]
            
            self.current_path = Path(file_info['image_path'])
            self.current_original = cv2.imread(str(self.current_path))
            
            if self.current_original is None:
                print(f"❌ 画像読み込み失敗: {self.current_path}")
                self.current_index += 1
                continue
            
            # 画像リサイズ（大きすぎる場合）
            height, width = self.current_original.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = 1200
                new_height = int(height * scale)
                self.current_image = cv2.resize(self.current_original, (new_width, new_height))
            else:
                self.current_image = self.current_original.copy()
            
            # マスク初期化
            self.mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            
            window_name = f"BCC Annotation [{self.current_index+1}/{len(self.bcc_files)}] - {self.current_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1000, 700)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            print(f"\n📸 [{self.current_index+1}/{len(self.bcc_files)}] {self.current_path.name}")
            print(f"画像サイズ: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
            print("BCC領域をマウスで描画してください...")
            
            while True:
                display_image = self.draw_overlay()
                
                # ブラシサイズを表示
                cv2.putText(display_image, f"Brush: {self.brush_size}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 進行状況を表示
                progress_text = f"[{self.current_index+1}/{len(self.bcc_files)}]"
                cv2.putText(display_image, progress_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.save_annotation(file_info['label_path']):
                        self.current_index += 1
                        break
                    else:
                        print("⚠️ BCC領域を描画してから保存してください")
                        
                elif key == ord('q'):  # スキップ
                    print("⏭️ スキップ")
                    self.current_index += 1
                    break
                    
                elif key == ord('r'):  # リセット
                    self.mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                    print("🔄 描画をリセット")
                    
                elif key == ord('h'):  # ヘルプ
                    self.show_help()
                    
                elif key == ord('+') or key == ord('='):  # ブラシサイズ拡大
                    self.brush_size = min(50, self.brush_size + 2)
                    print(f"🖌️ ブラシサイズ: {self.brush_size}")
                    
                elif key == ord('-'):  # ブラシサイズ縮小
                    self.brush_size = max(3, self.brush_size - 2)
                    print(f"🖌️ ブラシサイズ: {self.brush_size}")
                    
                elif key == 27:  # ESC
                    print("\n⭕ アノテーション終了")
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyWindow(window_name)
        
        print(f"\n🎉 BCC アノテーション完了!")
        print(f"✅ {len(self.bcc_files)}枚のBCC画像をアノテーション")

def main():
    print("🔬 BCC フリーハンドアノテーションツール")
    print("=" * 50)
    
    # BCCファイルの存在確認
    if not Path('bcc_additions.json').exists():
        print("❌ bcc_additions.json が見つかりません")
        print("先に add_bcc_only.py を実行してください")
        return
    
    annotator = BCCFreehandAnnotator()
    annotator.annotate_bcc_files()

if __name__ == "__main__":
    main()