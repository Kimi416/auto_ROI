#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
より大きなT字マスクを適用
"""

import cv2
import numpy as np
from pathlib import Path

def apply_larger_mask(filename):
    """より大きなT字マスクを適用"""
    
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
    
    # より大きなT字マスク（organized_advanced_maskedスタイル）
    # 眉毛から目の領域（横棒）- より大きく
    eyebrow_top = max(0, y - int(h * 0.4))  # より上から
    eye_bottom = y + int(h * 0.6)  # より下まで
    horizontal_left = x - int(w * 0.1)  # より左から
    horizontal_right = x + w + int(w * 0.1)  # より右まで
    
    # 鼻の領域 - より大きく
    nose_top = y + int(h * 0.2)
    nose_bottom = y + int(h * 0.8)
    nose_left = face_center_x - int(w * 0.2)
    nose_right = face_center_x + int(w * 0.2)
    
    # 口の領域 - より大きく
    mouth_top = y + int(h * 0.5)
    mouth_bottom = y + h + int(h * 0.1)
    mouth_left = face_center_x - int(w * 0.3)
    mouth_right = face_center_x + int(w * 0.3)
    
    # 画像境界内に制限
    height, width = image.shape[:2]
    horizontal_left = max(0, horizontal_left)
    horizontal_right = min(width, horizontal_right)
    eyebrow_top = max(0, eyebrow_top)
    eye_bottom = min(height, eye_bottom)
    
    nose_left = max(0, nose_left)
    nose_right = min(width, nose_right)
    nose_top = max(0, nose_top)
    nose_bottom = min(height, nose_bottom)
    
    mouth_left = max(0, mouth_left)
    mouth_right = min(width, mouth_right)
    mouth_top = max(0, mouth_top)
    mouth_bottom = min(height, mouth_bottom)
    
    # マスクを適用
    masked = image.copy()
    
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
    
    print(f"✅ 大きなT字マスク適用完了")
    print(f"📊 マスク後の黒領域比率: {black_ratio*100:.1f}%")
    
    return True

def main():
    problem_file = "Basal cell carcinoma_00011143.jpg"
    
    print("👤 大きなT字マスク適用")
    print("=" * 40)
    
    if apply_larger_mask(problem_file):
        print(f"\n🎉 {problem_file} の大きなマスク適用完了!")
    else:
        print(f"\n❌ {problem_file} の処理失敗")

if __name__ == "__main__":
    main()