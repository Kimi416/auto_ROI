#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
確実なT字マスクを適用（organized_advanced_maskedと同等レベル）
"""

import cv2
import numpy as np
from pathlib import Path

def apply_definitive_mask(filename):
    """確実なT字マスクを適用"""
    
    img_path = Path(f'organized_advanced_masked/Basal cell carcinoma/{filename}')
    
    print(f"📸 処理対象: {filename}")
    
    # 画像読み込み
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ 画像読み込み失敗")
        return False
    
    # 顔検出
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
    if len(faces) == 0:
        print(f"❌ 顔検出失敗")
        return False
    
    x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
    print(f"✅ 顔検出: ({x}, {y}, {w}, {h})")
    
    face_center_x = x + w // 2
    
    # organized_advanced_maskedスタイルの確実なT字マスク
    # 眉毛から目の領域（横棒）- 確実にカバー
    eyebrow_top = max(0, y - int(h * 0.5))  # 眉毛の上部（十分な余裕）
    eye_bottom = y + int(h * 0.65)  # 目の下部（十分な余裕）
    horizontal_left = max(0, x - int(w * 0.2))  # 顔の左端より広く
    horizontal_right = min(image.shape[1], x + w + int(w * 0.2))  # 顔の右端より広く
    
    # 鼻の領域 - より確実に
    nose_top = y + int(h * 0.15)
    nose_bottom = y + int(h * 0.85)
    nose_left = max(0, face_center_x - int(w * 0.25))
    nose_right = min(image.shape[1], face_center_x + int(w * 0.25))
    
    # 口の領域 - より確実に
    mouth_top = y + int(h * 0.45)
    mouth_bottom = min(image.shape[0], y + h + int(h * 0.15))
    mouth_left = max(0, face_center_x - int(w * 0.35))
    mouth_right = min(image.shape[1], face_center_x + int(w * 0.35))
    
    # マスクを適用
    masked = image.copy()
    
    print(f"横棒: ({horizontal_left}, {eyebrow_top}) - ({horizontal_right}, {eye_bottom})")
    print(f"鼻: ({nose_left}, {nose_top}) - ({nose_right}, {nose_bottom})")
    print(f"口: ({mouth_left}, {mouth_top}) - ({mouth_right}, {mouth_bottom})")
    
    # 横棒（眉毛から目）
    cv2.rectangle(masked, (horizontal_left, eyebrow_top), 
                 (horizontal_right, eye_bottom), (0, 0, 0), -1)
    
    # 鼻の領域
    cv2.rectangle(masked, (nose_left, nose_top), 
                 (nose_right, nose_bottom), (0, 0, 0), -1)
    
    # 口の領域
    cv2.rectangle(masked, (mouth_left, mouth_top), 
                 (mouth_right, mouth_bottom), (0, 0, 0), -1)
    
    # 保存
    cv2.imwrite(str(img_path), masked)
    
    # マスク後の黒領域比率を確認
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray_masked == 0)
    total_pixels = gray_masked.shape[0] * gray_masked.shape[1]
    black_ratio = black_pixels / total_pixels
    
    print(f"✅ 確実なT字マスク適用完了")
    print(f"📊 マスク後の黒領域比率: {black_ratio*100:.1f}%")
    
    return True

def main():
    problem_file = "Basal cell carcinoma_00011143.jpg"
    
    print("👤 確実なT字マスク適用")
    print("=" * 40)
    
    if apply_definitive_mask(problem_file):
        print(f"\n🎉 {problem_file} の確実なマスク適用完了!")
    else:
        print(f"\n❌ {problem_file} の処理失敗")

if __name__ == "__main__":
    main()