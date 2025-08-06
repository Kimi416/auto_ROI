#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特定の顔面画像のマスク修正
"""

import cv2
import numpy as np
from pathlib import Path

def fix_specific_face_image(filename):
    """特定の顔面画像にマスクを適用"""
    
    img_path = Path(f'organized_advanced_masked/Basal cell carcinoma/{filename}')
    
    if not img_path.exists():
        print(f"❌ ファイルが見つかりません: {img_path}")
        return False
    
    print(f"📸 処理対象: {filename}")
    
    # 画像読み込み
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ 画像読み込み失敗")
        return False
    
    # 顔検出器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 顔検出（複数のパラメータで試行）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_region = None
    
    for scale_factor in [1.05, 1.1, 1.2, 1.3]:
        for min_neighbors in [2, 3, 4, 5]:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors,
                minSize=(30, 30),
                maxSize=(400, 400)
            )
            if len(faces) > 0:
                # 最大の顔を選択
                face_region = max(faces, key=lambda x: x[2] * x[3])
                break
        if face_region is not None:
            break
    
    if face_region is None:
        print(f"❌ 顔検出失敗")
        return False
    
    x, y, w, h = face_region
    print(f"✅ 顔検出成功: ({x}, {y}, {w}, {h})")
    
    # 規定通りのT字マスクを適用
    face_center_x = x + w // 2
    
    # 眉毛から目の領域（横棒）
    eyebrow_top = max(0, y - int(h * 0.25))
    eye_bottom = y + int(h * 0.45)
    horizontal_left = x + int(w * 0.05)
    horizontal_right = x + w - int(w * 0.05)
    
    # 鼻の領域
    nose_top = y + int(h * 0.3)
    nose_bottom = y + int(h * 0.7)
    nose_left = face_center_x - int(w * 0.12)
    nose_right = face_center_x + int(w * 0.12)
    
    # 口の領域
    mouth_top = y + int(h * 0.6)
    mouth_bottom = y + h - int(h * 0.1)
    mouth_left = face_center_x - int(w * 0.18)
    mouth_right = face_center_x + int(w * 0.18)
    
    # マスクを適用
    masked = image.copy()
    
    # 横棒（眉毛から目）を黒く塗りつぶし
    cv2.rectangle(masked, (horizontal_left, eyebrow_top), 
                 (horizontal_right, eye_bottom), (0, 0, 0), -1)
    
    # 鼻の領域を黒く塗りつぶし
    cv2.rectangle(masked, (nose_left, nose_top), 
                 (nose_right, nose_bottom), (0, 0, 0), -1)
    
    # 口の領域を黒く塗りつぶし
    cv2.rectangle(masked, (mouth_left, mouth_top), 
                 (mouth_right, mouth_bottom), (0, 0, 0), -1)
    
    # 保存
    cv2.imwrite(str(img_path), masked)
    print(f"✅ T字マスク適用完了: {filename}")
    
    # マスク後の黒領域比率を確認
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray_masked == 0)
    total_pixels = gray_masked.shape[0] * gray_masked.shape[1]
    black_ratio = black_pixels / total_pixels
    print(f"📊 マスク後の黒領域比率: {black_ratio*100:.1f}%")
    
    return True

def main():
    print("👤 特定顔面画像マスク修正")
    print("=" * 40)
    
    # 問題のある顔面画像を修正
    problem_file = "Basal cell carcinoma_00011143.jpg"
    
    if fix_specific_face_image(problem_file):
        print(f"\n🎉 {problem_file} の修正完了!")
    else:
        print(f"\n❌ {problem_file} の修正失敗")

if __name__ == "__main__":
    main()