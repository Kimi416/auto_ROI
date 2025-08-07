#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大きな矩形マスクを適用するスクリプト
preview_adv_ADM_0035.jpgの赤い枠範囲と同じサイズでマスク適用
"""

import cv2
import numpy as np
from pathlib import Path

def apply_large_rect_mask(image_path, output_path=None):
    """大きな矩形マスクを適用"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return False
    
    img_h, img_w = image.shape[:2]
    
    # 大きな矩形マスク範囲（preview_adv_ADM_0035.jpgの赤い枠と同等）
    # 横方向: 画像の左から右まで広くカバー
    # 縦方向: 眉毛上から口下まで
    mask_left = int(img_w * 0.15)   # 左端
    mask_right = int(img_w * 0.85)  # 右端
    mask_top = int(img_h * 0.25)    # 上端（眉毛上）
    mask_bottom = int(img_h * 0.65) # 下端（口下）
    
    # マスクを適用
    masked_image = image.copy()
    cv2.rectangle(masked_image, 
                 (mask_left, mask_top), 
                 (mask_right, mask_bottom), 
                 (0, 0, 0), -1)  # 黒で塗りつぶし
    
    if output_path is None:
        output_path = Path(image_path).parent / f"large_masked_{Path(image_path).name}"
    
    cv2.imwrite(str(output_path), masked_image)
    print(f"大きな矩形マスク適用完了: {output_path}")
    
    return True

if __name__ == '__main__':
    # preview_adv_ADM_0035.jpgに大きな矩形マスクを適用
    image_path = "/Users/iinuma/Desktop/自動ROI/organized/ADM/preview_adv_ADM_0035.jpg"
    apply_large_rect_mask(image_path)