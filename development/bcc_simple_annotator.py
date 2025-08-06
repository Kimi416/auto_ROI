#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC簡易アノテーションツール
顔面・体幹部の区別なく、BCCアノテーションに集中
"""

import cv2
import numpy as np
import json
from pathlib import Path
import random

class BCCSimpleAnnotator:
    def __init__(self):
        self.bcc_class_id = 5
        self.current_image = None
        self.current_original = None
        self.current_path = None
        self.polygons = []
        self.current_polygon = []
        self.progress_file = 'bcc_annotation_progress.json'
        
        # BCCファイル一覧を読み込み
        with open('bcc_additions.json', 'r', encoding='utf-8') as f:
            self.bcc_files = json.load(f)
        
        # 進行状況を読み込み
        self.load_progress()
        
    def load_progress(self):
        """進行状況を読み込み"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                self.current_index = progress.get('current_index', 0)
                completed_files = set(progress.get('completed_files', []))
                
                # 完了済みファイルをフィルタリング
                self.bcc_files = [f for f in self.bcc_files if f['image_path'] not in completed_files]
                
                print(f"📂 進行状況を復元: {self.current_index}枚完了済み")
                print(f"📁 残り: {len(self.bcc_files)}枚")
                
            except Exception as e:
                print(f"⚠️ 進行状況の読み込みエラー: {e}")
                self.current_index = 0
        else:
            self.current_index = 0
            # ランダムシャッフル
            random.shuffle(self.bcc_files)
    
    def save_progress(self, completed_file_path):
        """進行状況を保存"""
        try:
            progress = {'current_index': self.current_index, 'completed_files': []}
            
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    existing_progress = json.load(f)
                progress['completed_files'] = existing_progress.get('completed_files', [])
            
            if completed_file_path not in progress['completed_files']:
                progress['completed_files'].append(completed_file_path)
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ 進行状況の保存エラー: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
            print(f"📍 頂点追加: ({x}, {y}) [合計: {len(self.current_polygon)}点]")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
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
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
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
    
    def save_annotation(self, label_path, image_path):
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
        
        self.save_progress(image_path)
        print(f"✅ {len(bboxes)}個のBCCを保存: {Path(label_path).name}")
        return True
    
    def draw_overlay(self):
        """ポリゴンを重ねた画像を作成"""
        overlay = self.current_image.copy()
        
        # 完成したポリゴンを描画
        for i, polygon in enumerate(self.polygons):
            if len(polygon) >= 3:
                pts = np.array(polygon, np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0, 100))
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
                
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(overlay, f'BCC{i+1}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 現在描画中のポリゴンを描画
        if len(self.current_polygon) > 0:
            for point in self.current_polygon:
                cv2.circle(overlay, tuple(point), 5, (0, 0, 255), -1)
            
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, np.int32)
                cv2.polylines(overlay, [pts], False, (0, 0, 255), 2)
            
            if len(self.current_polygon) >= 3:
                cv2.line(overlay, tuple(self.current_polygon[-1]), tuple(self.current_polygon[0]), (255, 0, 0), 1)
        
        result = cv2.addWeighted(self.current_image, 0.7, overlay, 0.3, 0)
        return result
    
    def show_help(self):
        """ヘルプを表示"""
        print("\n" + "="*60)
        print("🖱️  BCC 簡易アノテーション")
        print("="*60)
        print("⚠️  注意: 顔面画像は事前に目・鼻・口をマスクしてください")  
        print("")
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
        print("  ESC : 安全に終了（進行状況保存）")
        print("")
        print("💡 顔面画像の処理:")
        print("1. 顔が写っている画像は'q'でスキップ")
        print("2. 後で mask.py で顔面マスク処理")
        print("3. 体幹部画像のみアノテーション実行")
        print("="*60)
    
    def annotate_bcc_files(self):
        """BCCファイルのアノテーション実行"""
        if not self.bcc_files:
            print("🎉 全てのBCCアノテーションが完了済みです！")
            return
            
        print(f"🔬 BCC 簡易アノテーション開始")
        print(f"📁 残りファイル数: {len(self.bcc_files)}枚")
        print(f"⚠️  顔面画像は'q'でスキップしてください")
        
        self.show_help()
        
        file_index = 0
        while file_index < len(self.bcc_files):
            file_info = self.bcc_files[file_index]
            
            self.current_path = Path(file_info['image_path'])
            self.current_original = cv2.imread(str(self.current_path))
            
            if self.current_original is None:
                print(f"❌ 画像読み込み失敗: {self.current_path}")
                file_index += 1
                continue
            
            # 画像リサイズ
            height, width = self.current_original.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = 1200
                new_height = int(height * scale)
                self.current_image = cv2.resize(self.current_original, (new_width, new_height))
            else:
                self.current_image = self.current_original.copy()
            
            # 初期化
            self.polygons = []
            self.current_polygon = []
            
            window_name = f"BCC Annotation [{file_index+1}/{len(self.bcc_files)}] - {self.current_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1000, 700)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            total_done = self.current_index + file_index
            print(f"\n📸 [完了: {total_done}, 残り: {len(self.bcc_files) - file_index}] {self.current_path.name}")
            print("🎯 体幹部BCC: 左クリックで囲む, 右クリックで完成")
            print("👤 顔面画像: 'q'でスキップ")
            
            while True:
                display_image = self.draw_overlay()
                
                # 進行状況表示
                progress_text = f"Done: {total_done}, Left: {len(self.bcc_files) - file_index}"
                cv2.putText(display_image, progress_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 警告表示
                warning_text = "Face images: Press 'q' to skip"
                cv2.putText(display_image, warning_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                bcc_text = f"BCC: {len(self.polygons)}"
                cv2.putText(display_image, bcc_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.current_polygon:
                    vertex_text = f"Points: {len(self.current_polygon)}"
                    cv2.putText(display_image, vertex_text, (10, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.save_annotation(file_info['label_path'], file_info['image_path']):
                        file_index += 1
                        break
                    else:
                        print("⚠️ BCC領域を囲んでから保存してください")
                        
                elif key == ord('q'):  # スキップ
                    print("⏭️ スキップ（顔面画像?）")
                    self.save_progress(file_info['image_path'])
                    file_index += 1
                    break
                    
                elif key == ord('r'):  # 全リセット
                    self.polygons = []
                    self.current_polygon = []
                    print("🔄 全てリセット")
                    
                elif key == ord('u'):  # 最後のポリゴン削除
                    if self.polygons:
                        removed = self.polygons.pop()
                        print(f"🗑️ 最後のBCCポリゴンを削除 ({len(removed)}点)")
                        
                elif key == ord('c'):  # 現在のポリゴンキャンセル
                    if self.current_polygon:
                        print(f"❌ 現在のポリゴンをキャンセル ({len(self.current_polygon)}点)")
                        self.current_polygon = []
                    
                elif key == ord('h'):  # ヘルプ
                    self.show_help()
                    
                elif key == 27:  # ESC - 安全な終了
                    print(f"\n💾 進行状況を保存して終了...")
                    print(f"📊 完了: {total_done}枚")
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyWindow(window_name)
        
        print(f"\n🎉 BCC アノテーション完了!")
        print(f"✅ 総完了数: {self.current_index + len(self.bcc_files)}枚")
        print(f"\n💡 次のステップ:")
        print(f"1. 顔面画像のマスク処理（必要に応じて）")
        print(f"2. 学習開始")
        
        # 完了時は進行状況ファイルを削除
        if Path(self.progress_file).exists():
            Path(self.progress_file).unlink()
            print("📁 進行状況ファイルを削除（完了のため）")

def main():
    print("🔬 BCC 簡易アノテーションツール")
    print("=" * 50)
    print("⚠️  顔面画像は'q'でスキップしてください")
    
    if not Path('bcc_additions.json').exists():
        print("❌ bcc_additions.json が見つかりません")
        return
    
    annotator = BCCSimpleAnnotator()
    annotator.annotate_bcc_files()

if __name__ == "__main__":
    main()