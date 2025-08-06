#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC専用ポリゴンアノテーションツール
クリックで頂点を指定してBCC領域を囲む
"""

import cv2
import numpy as np
import json
from pathlib import Path
import random

class BCCPolygonAnnotator:
    def __init__(self):
        self.bcc_class_id = 5
        self.current_image = None
        self.current_original = None
        self.current_path = None
        self.polygons = []  # 複数のポリゴンを保存
        self.current_polygon = []  # 現在描画中のポリゴン
        self.drawing = False
        
        # BCCファイル一覧を読み込み
        with open('bcc_additions.json', 'r', encoding='utf-8') as f:
            self.bcc_files = json.load(f)
        
        # ランダムシャッフルで変化をつける
        random.shuffle(self.bcc_files)
        self.current_index = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左クリックで頂点追加
            self.current_polygon.append([x, y])
            print(f"📍 頂点追加: ({x}, {y}) [合計: {len(self.current_polygon)}点]")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右クリックでポリゴン完成
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                print(f"✅ ポリゴン完成: {len(self.current_polygon)}点 [BCC数: {len(self.polygons)}]")
                self.current_polygon = []
            else:
                print("⚠️ 最低3点必要です")
    
    def create_bbox_from_polygons(self):
        """ポリゴンからバウンディングボックスを作成"""
        bboxes = []
        
        for polygon in self.polygons:
            if len(polygon) < 3:
                continue
                
            # ポリゴンの境界ボックスを計算
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            # 最小サイズチェック
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                bboxes.append([x1, y1, x2, y2])
        
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
        bboxes = self.create_bbox_from_polygons()
        
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
        """ポリゴンを重ねた画像を作成"""
        overlay = self.current_image.copy()
        
        # 完成したポリゴンを描画
        for i, polygon in enumerate(self.polygons):
            if len(polygon) >= 3:
                # ポリゴンを塗りつぶし
                pts = np.array(polygon, np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0, 100))
                
                # ポリゴンの輪郭を描画
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
                
                # バウンディングボックスも表示
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(overlay, f'BCC{i+1}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 現在描画中のポリゴンを描画
        if len(self.current_polygon) > 0:
            # 頂点を描画
            for point in self.current_polygon:
                cv2.circle(overlay, tuple(point), 5, (0, 0, 255), -1)
            
            # 線を描画
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, np.int32)
                cv2.polylines(overlay, [pts], False, (0, 0, 255), 2)
            
            # 現在の点と最初の点を結ぶ線（プレビュー）
            if len(self.current_polygon) >= 3:
                cv2.line(overlay, tuple(self.current_polygon[-1]), tuple(self.current_polygon[0]), (255, 0, 0), 1)
        
        # 半透明で重ね合わせ
        result = cv2.addWeighted(self.current_image, 0.7, overlay, 0.3, 0)
        
        return result
    
    def show_help(self):
        """ヘルプを表示"""
        print("\n" + "="*60)
        print("🖱️  BCC ポリゴンアノテーション操作方法")
        print("="*60)
        print("左クリック       : 頂点を追加")
        print("右クリック       : ポリゴン完成（3点以上必要）")
        print("")
        print("キーボード操作:")
        print("  's' : 保存して次へ")
        print("  'q' : スキップして次へ")  
        print("  'r' : 全て削除してリセット")
        print("  'u' : 最後のポリゴンを削除")
        print("  'c' : 現在描画中のポリゴンをキャンセル")
        print("  'h' : ヘルプ表示")
        print("  ESC : 終了")
        print("")
        print("💡 使い方:")
        print("1. BCC領域の周りを左クリックで囲む")
        print("2. 右クリックでポリゴン完成")
        print("3. 複数のBCCがある場合は繰り返し")
        print("4. 's'で保存して次の画像へ")
        print("="*60)
    
    def annotate_bcc_files(self):
        """BCCファイルのアノテーション実行"""
        print(f"🔬 BCC ポリゴンアノテーション開始")
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
            
            # ポリゴン初期化
            self.polygons = []
            self.current_polygon = []
            
            window_name = f"BCC Annotation [{self.current_index+1}/{len(self.bcc_files)}] - {self.current_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1000, 700)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            print(f"\n📸 [{self.current_index+1}/{len(self.bcc_files)}] {self.current_path.name}")
            print(f"画像サイズ: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
            print("BCC領域を左クリックで囲み、右クリックで完成させてください...")
            
            while True:
                display_image = self.draw_overlay()
                
                # 進行状況を表示
                progress_text = f"[{self.current_index+1}/{len(self.bcc_files)}]"
                cv2.putText(display_image, progress_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # BCC数を表示
                bcc_text = f"BCC: {len(self.polygons)}"
                cv2.putText(display_image, bcc_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 現在のポリゴンの頂点数を表示
                if self.current_polygon:
                    vertex_text = f"Points: {len(self.current_polygon)}"
                    cv2.putText(display_image, vertex_text, (10, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.save_annotation(file_info['label_path']):
                        self.current_index += 1
                        break
                    else:
                        print("⚠️ BCC領域を囲んでから保存してください")
                        
                elif key == ord('q'):  # スキップ
                    print("⏭️ スキップ")
                    self.current_index += 1
                    break
                    
                elif key == ord('r'):  # 全リセット
                    self.polygons = []
                    self.current_polygon = []
                    print("🔄 全てリセット")
                    
                elif key == ord('u'):  # 最後のポリゴン削除
                    if self.polygons:
                        removed = self.polygons.pop()
                        print(f"🗑️ 最後のBCCポリゴンを削除 ({len(removed)}点)")
                    else:
                        print("⚠️ 削除するポリゴンがありません")
                        
                elif key == ord('c'):  # 現在のポリゴンキャンセル
                    if self.current_polygon:
                        print(f"❌ 現在のポリゴンをキャンセル ({len(self.current_polygon)}点)")
                        self.current_polygon = []
                    
                elif key == ord('h'):  # ヘルプ
                    self.show_help()
                    
                elif key == 27:  # ESC
                    print("\n⭕ アノテーション終了")
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyWindow(window_name)
        
        print(f"\n🎉 BCC アノテーション完了!")
        print(f"✅ {len(self.bcc_files)}枚のBCC画像をアノテーション")

def main():
    print("🔬 BCC ポリゴンアノテーションツール")
    print("=" * 50)
    
    # BCCファイルの存在確認
    if not Path('bcc_additions.json').exists():
        print("❌ bcc_additions.json が見つかりません")
        print("先に add_bcc_only.py を実行してください")
        return
    
    annotator = BCCPolygonAnnotator()
    annotator.annotate_bcc_files()

if __name__ == "__main__":
    main()