#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
organized_advanced_masked フォルダ用アノテーションツール
マスク済み画像に対応したBCCアノテーションシステム
"""

import cv2
import numpy as np
import json
from pathlib import Path
import random
import os

class OrganizedMaskedAnnotator:
    def __init__(self):
        self.class_mapping = {
            'ADM': 0,
            'Ephelis': 1,
            'Melasma': 2,
            'Solar lentigo': 3,
            'Nevus': 4,
            'Basal cell carcinoma': 5,
            'Seborrheic keratosis': 6,
            'Malignant melanoma': 7
        }
        
        self.current_image = None
        self.current_original = None
        self.current_path = None
        self.current_class_id = None
        self.polygons = []
        self.current_polygon = []
        self.progress_file = 'organized_masked_progress.json'
        
        # organized_advanced_masked フォルダからファイルリストを作成
        self.create_file_list()
        
        # 進行状況を読み込み
        self.load_progress()
        
    def create_file_list(self):
        """organized_advanced_maskedフォルダからファイルリストを作成"""
        self.image_files = []
        base_dir = Path('organized_advanced_masked')
        
        if not base_dir.exists():
            print("❌ organized_advanced_masked フォルダが見つかりません")
            return
        
        print("📁 organized_advanced_masked フォルダを検索中...")
        
        for class_name, class_id in self.class_mapping.items():
            class_dir = base_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
                
                for img_path in image_files:
                    # ラベルファイルパスを決定
                    label_path = img_path.with_suffix('.txt')
                    
                    self.image_files.append({
                        'image_path': str(img_path),
                        'label_path': str(label_path), 
                        'class_name': class_name,
                        'class_id': class_id
                    })
                
                print(f"  {class_name}: {len(image_files)}枚")
        
        print(f"📊 総画像数: {len(self.image_files)}枚")
        
        # BCC画像を優先的に処理するため先頭に移動
        bcc_files = [f for f in self.image_files if f['class_name'] == 'Basal cell carcinoma']
        other_files = [f for f in self.image_files if f['class_name'] != 'Basal cell carcinoma']
        
        # BCCをランダムシャッフルして先頭に
        random.shuffle(bcc_files)
        random.shuffle(other_files)
        
        self.image_files = bcc_files + other_files
        print(f"🎯 BCC画像を優先: {len(bcc_files)}枚のBCC画像を先頭に配置")
    
    def load_progress(self):
        """進行状況を読み込み"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                self.current_index = progress.get('current_index', 0)
                completed_files = set(progress.get('completed_files', []))
                
                # 完了済みファイルをフィルタリング
                self.image_files = [f for f in self.image_files if f['image_path'] not in completed_files]
                
                print(f"📂 進行状況を復元: {self.current_index}枚完了済み")
                print(f"📁 残り: {len(self.image_files)}枚")
                
            except Exception as e:
                print(f"⚠️ 進行状況の読み込みエラー: {e}")
                self.current_index = 0
        else:
            self.current_index = 0
    
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
                print(f"✅ ポリゴン完成: {len(self.current_polygon)}点 [病変数: {len(self.polygons)}]")
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
    
    def save_annotation(self, file_info):
        """アノテーションを保存"""
        bboxes = self.create_bbox_from_polygons()
        
        if not bboxes:
            print("⚠️ 病変アノテーションがありません")
            return False
        
        img_height, img_width = self.current_image.shape[:2]
        
        # 元の画像サイズでスケール調整が必要な場合
        if self.current_image.shape != self.current_original.shape:
            orig_height, orig_width = self.current_original.shape[:2]
            scale_x = orig_width / img_width
            scale_y = orig_height / img_height
            
            # バウンディングボックスを元の画像サイズにスケール
            scaled_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                scaled_bbox = [
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y)
                ]
                scaled_bboxes.append(scaled_bbox)
            bboxes = scaled_bboxes
            img_width, img_height = orig_width, orig_height
        
        label_path = file_info['label_path']
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                center_x, center_y, width, height = self.convert_to_yolo(bbox, img_width, img_height)
                f.write(f"{file_info['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        self.save_progress(file_info['image_path'])
        print(f"✅ {len(bboxes)}個の{file_info['class_name']}を保存: {Path(label_path).name}")
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
                cv2.putText(overlay, f'Lesion{i+1}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
        print("🖱️  organized_advanced_masked アノテーション")
        print("="*60)
        print("✅ マスク済み画像対応")  
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
        print("💡 BCC画像が優先的に表示されます")
        print("🎭 顔面画像は既にマスク処理済みです")
        print("="*60)
    
    def annotate_images(self):
        """画像のアノテーション実行"""
        if not self.image_files:
            print("🎉 全てのアノテーションが完了済みです！")
            return
            
        print(f"🔬 organized_advanced_masked アノテーション開始")
        print(f"📁 残りファイル数: {len(self.image_files)}枚")
        
        self.show_help()
        
        file_index = 0
        while file_index < len(self.image_files):
            file_info = self.image_files[file_index]
            
            self.current_path = Path(file_info['image_path'])
            self.current_class_id = file_info['class_id']
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
            
            window_name = f"{file_info['class_name']} [{file_index+1}/{len(self.image_files)}] - {self.current_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1000, 700)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            total_done = self.current_index + file_index
            print(f"\n📸 [完了: {total_done}, 残り: {len(self.image_files) - file_index}] {file_info['class_name']}: {self.current_path.name}")
            
            # BCCかどうかで表示メッセージを変更
            if file_info['class_name'] == 'Basal cell carcinoma':
                print("🎯 BCC画像: 左クリックで囲む, 右クリックで完成")
                print("🎭 顔面は既にマスク済み - 安心してアノテーション可能")
            else:
                print("🔬 病変領域: 左クリックで囲む, 右クリックで完成")
            
            while True:
                display_image = self.draw_overlay()
                
                # 進行状況表示
                progress_text = f"Done: {total_done}, Left: {len(self.image_files) - file_index}"
                cv2.putText(display_image, progress_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # クラス名表示
                class_text = f"Class: {file_info['class_name']}"
                cv2.putText(display_image, class_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                lesion_text = f"Lesions: {len(self.polygons)}"
                cv2.putText(display_image, lesion_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.current_polygon:
                    vertex_text = f"Points: {len(self.current_polygon)}"
                    cv2.putText(display_image, vertex_text, (10, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(window_name, display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    if self.save_annotation(file_info):
                        file_index += 1
                        break
                    else:
                        print("⚠️ 病変領域を囲んでから保存してください")
                        
                elif key == ord('q'):  # スキップ
                    print("⏭️ スキップ")
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
                        print(f"🗑️ 最後のポリゴンを削除 ({len(removed)}点)")
                        
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
        
        print(f"\n🎉 全アノテーション完了!")
        print(f"✅ 総完了数: {self.current_index + len(self.image_files)}枚")
        print(f"\n💡 次のステップ:")
        print(f"1. データセット構造の確認")
        print(f"2. YOLO学習の開始")
        
        # 完了時は進行状況ファイルを削除
        if Path(self.progress_file).exists():
            Path(self.progress_file).unlink()
            print("📁 進行状況ファイルを削除（完了のため）")

def main():
    print("🔬 organized_advanced_masked アノテーションツール")
    print("=" * 50)
    print("🎭 マスク済み画像対応アノテーションシステム")
    
    annotator = OrganizedMaskedAnnotator()
    
    if not annotator.image_files:
        print("❌ アノテーション対象ファイルが見つかりません")
        return
    
    annotator.annotate_images()

if __name__ == "__main__":
    main()