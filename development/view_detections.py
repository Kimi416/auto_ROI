#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
検出結果ビューア
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
matplotlib.rcParams['font.family'] = 'sans-serif'

def display_detection_results(result_dir='batch_results'):
    """検出結果画像を表示"""
    result_path = Path(result_dir)
    
    # 検出結果画像を収集
    detection_images = list(result_path.glob('*_detection.jpg'))
    
    if not detection_images:
        print(f"⚠️ {result_dir}に検出結果画像が見つかりません")
        return
    
    # 表示
    n_images = len(detection_images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    
    # 1つの画像の場合
    if n_images == 1:
        axes = [axes]
    # 複数行の場合
    elif rows > 1:
        axes = axes.flatten()
    
    for i, img_path in enumerate(detection_images):
        img = mpimg.imread(str(img_path))
        axes[i].imshow(img)
        axes[i].set_title(img_path.stem.replace('_detection', ''))
        axes[i].axis('off')
    
    # 余った軸を非表示
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'YOLO病変検出結果 ({n_images}枚)', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='検出結果ビューア')
    parser.add_argument('-d', '--dir', default='batch_results', help='結果ディレクトリ')
    
    args = parser.parse_args()
    
    display_detection_results(args.dir)

if __name__ == "__main__":
    main()