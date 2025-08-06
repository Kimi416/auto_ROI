#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCCデータのみを既存データセットに追加
既存のアノテーション済みデータを保持して、BCCのみアノテーション
"""

import os
import json
import shutil
from pathlib import Path
import random

def add_bcc_to_existing_dataset():
    """既存データセットにBCCデータのみ追加"""
    print("🔬 既存データセットにBCCデータ追加")
    
    # 既存のyolo_datasetをバックアップ
    existing_dataset = Path('yolo_dataset')
    if not existing_dataset.exists():
        print("❌ 既存のyolo_datasetが見つかりません")
        return False
    
    # バックアップ作成
    backup_dataset = Path('yolo_dataset_backup')
    if backup_dataset.exists():
        shutil.rmtree(backup_dataset)
    shutil.copytree(existing_dataset, backup_dataset)
    print(f"✅ バックアップ作成: {backup_dataset}")
    
    # BCCデータを取得
    bcc_dir = Path('organized/Basal cell carcinoma')
    if not bcc_dir.exists():
        print("❌ BCC画像フォルダが見つかりません")
        return False
    
    # BCC画像ファイル取得
    bcc_images = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.bmp', '*.tif']:
        bcc_images.extend(list(bcc_dir.glob(ext)))
    
    print(f"📁 BCC画像数: {len(bcc_images)}枚")
    
    # データ分割（既存と同じ比率: train:70%, valid:20%, test:10%）
    random.shuffle(bcc_images)
    train_count = int(len(bcc_images) * 0.7)
    valid_count = int(len(bcc_images) * 0.2)
    
    splits = {
        'train': bcc_images[:train_count],
        'valid': bcc_images[train_count:train_count + valid_count],
        'test': bcc_images[train_count + valid_count:]
    }
    
    bcc_class_id = 5  # Basal cell carcinomaのクラスID
    added_files = []
    
    print("\n📂 BCCデータを各分割に追加:")
    
    for split, files in splits.items():
        print(f"  {split}: {len(files)}枚を追加中...")
        
        split_dir = existing_dataset / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        for i, img_path in enumerate(files):
            # 新しいファイル名
            new_name = f"BCC_{i:04d}_{split}.jpg"
            dst_img = images_dir / new_name
            label_file = labels_dir / f"BCC_{i:04d}_{split}.txt"
            
            try:
                # 画像をコピー
                shutil.copy2(img_path, dst_img)
                
                # 仮のラベルファイル作成（要アノテーション）
                with open(label_file, 'w') as f:
                    # 仮の中央位置BCC
                    f.write(f"{bcc_class_id} 0.5 0.5 0.3 0.3\n")
                
                added_files.append({
                    'image_path': str(dst_img),
                    'label_path': str(label_file),
                    'original_path': str(img_path),
                    'split': split,
                    'needs_annotation': True
                })
                
            except Exception as e:
                print(f"    エラー: {img_path} → {e}")
    
    # 追加情報を保存
    with open('bcc_additions.json', 'w', encoding='utf-8') as f:
        json.dump(added_files, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ BCC追加完了: {len(added_files)}ファイル")
    print(f"  Train: {len(splits['train'])}枚")
    print(f"  Valid: {len(splits['valid'])}枚") 
    print(f"  Test: {len(splits['test'])}枚")
    
    return True

def create_bcc_annotation_tool():
    """BCC専用アノテーションツール作成"""
    print("\n🔧 BCC専用アノテーションツール作成")
    
    tool_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC専用アノテーションツール
追加されたBCCファイルのみをアノテーション
"""

import cv2
import numpy as np
import json
from pathlib import Path

class BCCOnlyAnnotator:
    def __init__(self):
        self.bcc_class_id = 5
        self.current_image = None
        self.current_path = None
        self.bboxes = []
        self.drawing = False
        self.bbox = []
        
        # BCCファイル一覧を読み込み
        with open('bcc_additions.json', 'r', encoding='utf-8') as f:
            self.bcc_files = json.load(f)
        
        self.current_index = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.bbox[2] = x
            self.bbox[3] = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if abs(self.bbox[2] - self.bbox[0]) > 10 and abs(self.bbox[3] - self.bbox[1]) > 10:
                self.bboxes.append(self.bbox.copy())
    
    def draw_bboxes(self):
        img = self.current_image.copy()
        for bbox in self.bboxes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, 'BCC', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.drawing and len(self.bbox) == 4:
            cv2.rectangle(img, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0, 0, 255), 2)
        
        return img
    
    def convert_to_yolo(self, bbox, img_width, img_height):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return center_x, center_y, width, height
    
    def save_annotation(self, label_path):
        if not self.bboxes:
            print("⚠️ BCCアノテーションがありません")
            return
        
        img_height, img_width = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for bbox in self.bboxes:
                center_x, center_y, width, height = self.convert_to_yolo(bbox, img_width, img_height)
                f.write(f"{self.bcc_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\\n")
        
        print(f"✅ {len(self.bboxes)}個のBCCを保存: {label_path}")
    
    def annotate_bcc_files(self):
        print(f"🔬 BCC専用アノテーション開始 ({len(self.bcc_files)}ファイル)")
        
        while self.current_index < len(self.bcc_files):
            file_info = self.bcc_files[self.current_index]
            
            self.current_path = Path(file_info['image_path'])
            self.current_image = cv2.imread(str(self.current_path))
            
            if self.current_image is None:
                print(f"❌ 画像読み込み失敗: {self.current_path}")
                self.current_index += 1
                continue
            
            self.bboxes = []
            
            window_name = f"BCC Annotation [{self.current_index+1}/{len(self.bcc_files)}] - {self.current_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            print(f"\\n📸 [{self.current_index+1}/{len(self.bcc_files)}] {self.current_path.name}")
            print("操作: ドラッグでBCC選択, 's'=保存, 'u'=削除, 'r'=リセット, 'q'=スキップ, ESC=終了")
            
            while True:
                img_display = self.draw_bboxes()
                cv2.imshow(window_name, img_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # 保存
                    self.save_annotation(file_info['label_path'])
                    self.current_index += 1
                    break
                elif key == ord('u'):  # 削除
                    if self.bboxes:
                        self.bboxes.pop()
                elif key == ord('r'):  # リセット
                    self.bboxes = []
                elif key == ord('q'):  # スキップ
                    self.current_index += 1
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyWindow(window_name)
        
        print("\\n🎉 BCC専用アノテーション完了!")

if __name__ == "__main__":
    annotator = BCCOnlyAnnotator()
    annotator.annotate_bcc_files()
'''
    
    with open('bcc_only_annotator.py', 'w', encoding='utf-8') as f:
        f.write(tool_code)
    
    print("✅ bcc_only_annotator.py 作成完了")

def main():
    print("🔬 BCC追加学習用データセット準備")
    print("=" * 50)
    print("既存データ保持 + BCC追加のみ")
    
    # 1. BCCデータを既存データセットに追加
    if not add_bcc_to_existing_dataset():
        return
    
    # 2. BCC専用アノテーションツール作成
    create_bcc_annotation_tool()
    
    print(f"\n🎉 BCC追加準備完了!")
    print("\n次のステップ:")
    print("1. python3 bcc_only_annotator.py でBCCのみアノテーション")
    print("2. 既存モデルで追加学習実行")
    print("\n💡 メリット:")
    print("- 既存の407枚のアノテーションを保持")
    print("- BCCのみ1193枚を新規アノテーション")
    print("- 大幅な不均衡解消（3→1193枚）")

if __name__ == "__main__":
    random.seed(42)
    main()