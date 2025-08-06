#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC追加画像の顔面マスク処理
追加されたBCC画像の中で顔面のものに目・鼻・口のT字マスクを適用
"""

import cv2
import numpy as np
import json
from pathlib import Path
import os

class BCCFaceMasker:
    def __init__(self):
        # OpenCVの顔検出器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # BCCファイル一覧を読み込み
        with open('bcc_additions.json', 'r', encoding='utf-8') as f:
            self.bcc_files = json.load(f)
        
        self.face_files = []
        self.trunk_files = []
        self.processed_count = 0
        
    def detect_face_features(self, image):
        """顔の特徴を検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, False
        
        face_features = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # 目検出
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            # 特徴を絶対座標に変換
            absolute_eyes = []
            for (ex, ey, ew, eh) in eyes:
                absolute_eyes.append((x + ex, y + ey, ew, eh))
            
            # 鼻と口の推定位置
            nose_y = y + int(h * 0.4)
            nose_h = int(h * 0.25)
            mouth_y = y + int(h * 0.65)
            mouth_h = int(h * 0.2)
            
            face_features.append({
                'face': (x, y, w, h),
                'eyes': absolute_eyes,
                'nose': (x + int(w * 0.25), nose_y, int(w * 0.5), nose_h),
                'mouth': (x + int(w * 0.2), mouth_y, int(w * 0.6), mouth_h)
            })
        
        return face_features, True
    
    def apply_t_mask(self, image, face_features):
        """T字マスクを適用"""
        masked = image.copy()
        
        for features in face_features:
            # 目のマスク
            for (ex, ey, ew, eh) in features['eyes']:
                # 目の周りを少し大きめにマスク
                margin = 5
                cv2.rectangle(masked, 
                            (ex - margin, ey - margin), 
                            (ex + ew + margin, ey + eh + margin), 
                            (0, 0, 0), -1)
            
            # 鼻のマスク
            nx, ny, nw, nh = features['nose']
            cv2.rectangle(masked, (nx, ny), (nx + nw, ny + nh), (0, 0, 0), -1)
            
            # 口のマスク
            mx, my, mw, mh = features['mouth']
            cv2.rectangle(masked, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)
        
        return masked
    
    def classify_and_process_images(self):
        """画像を分類して顔面のものをマスク処理"""
        print(f"🔍 BCC画像の分類とマスク処理開始")
        print(f"📁 対象画像数: {len(self.bcc_files)}枚")
        
        for i, file_info in enumerate(self.bcc_files):
            img_path = Path(file_info['image_path'])
            
            print(f"\n📸 [{i+1}/{len(self.bcc_files)}] {img_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ 画像読み込み失敗: {img_path}")
                continue
            
            # 顔検出
            face_features, is_face = self.detect_face_features(image)
            
            if is_face and face_features:
                print(f"👤 顔面画像: {len(face_features)}個の顔を検出")
                
                # マスク適用
                masked_image = self.apply_t_mask(image, face_features)
                
                # 元の画像を上書き
                cv2.imwrite(str(img_path), masked_image)
                
                self.face_files.append(file_info)
                self.processed_count += 1
                
                print(f"✅ T字マスク適用完了: {img_path.name}")
                
            else:
                print(f"🫁 体幹部画像: マスク不要")
                self.trunk_files.append(file_info)
        
        # 結果保存
        result = {
            'face_files': self.face_files,
            'trunk_files': self.trunk_files,
            'processed_count': self.processed_count,
            'total_files': len(self.bcc_files)
        }
        
        with open('bcc_face_mask_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 処理結果:")
        print(f"  顔面画像（マスク済み）: {len(self.face_files)}枚")
        print(f"  体幹部画像: {len(self.trunk_files)}枚")
        print(f"  マスク処理済み: {self.processed_count}枚")
        print(f"✅ 結果保存: bcc_face_mask_result.json")
    
    def preview_results(self):
        """処理結果のプレビュー表示"""
        if not self.face_files:
            print("👤 顔面画像が見つかりませんでした")
            return
        
        print(f"\n🔍 マスク結果プレビュー（最初の3枚）")
        
        for i, file_info in enumerate(self.face_files[:3]):
            img_path = Path(file_info['image_path'])
            image = cv2.imread(str(img_path))
            
            if image is not None:
                # リサイズして表示
                height, width = image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = 800
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                window_name = f"Masked BCC Face {i+1} - {img_path.name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, image)
                
                print(f"📸 プレビュー表示中: {img_path.name}")
                print("何かキーを押すと次へ...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

def main():
    print("🎭 BCC顔面画像マスク処理")
    print("=" * 50)
    print("追加されたBCC画像の顔面検出とT字マスク適用")
    
    if not Path('bcc_additions.json').exists():
        print("❌ bcc_additions.json が見つかりません")
        print("先に add_bcc_only.py を実行してください")
        return
    
    masker = BCCFaceMasker()
    
    # 分類とマスク処理
    masker.classify_and_process_images()
    
    # プレビュー表示
    print(f"\nプレビューを表示しますか？ (y/n): ", end="")
    try:
        response = input().lower()
        if response == 'y':
            masker.preview_results()
    except:
        pass
    
    print(f"\n🎉 BCC顔面マスク処理完了!")
    print(f"💡 次のステップ:")
    print(f"1. python3 bcc_simple_annotator.py でアノテーション開始")
    print(f"2. マスク済み顔面画像も安心してアノテーション可能")

if __name__ == "__main__":
    main()