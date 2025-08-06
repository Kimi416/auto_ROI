#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
切り抜き病変プレビューア
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
matplotlib.rcParams['font.family'] = 'sans-serif'

def preview_extracted_lesions(lesion_dir='extracted_lesions', max_images=12):
    """切り抜かれた病変画像をプレビュー"""
    lesion_path = Path(lesion_dir)
    
    # PNG画像を収集（最新のものから）
    lesion_images = sorted(lesion_path.glob('*.png'), key=lambda x: x.stat().st_mtime, reverse=True)
    lesion_images = lesion_images[:max_images]
    
    if not lesion_images:
        print(f"⚠️ {lesion_dir}に切り抜き画像が見つかりません")
        return
    
    # 表示
    n_images = len(lesion_images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # 1つの画像の場合
    if n_images == 1:
        axes = [axes]
    # 複数行の場合
    elif rows > 1:
        axes = axes.flatten()
    # 1行の場合
    elif cols > 1:
        axes = list(axes)
    else:
        axes = [axes]
    
    for i, img_path in enumerate(lesion_images):
        # 透過PNG読み込み
        img = Image.open(img_path)
        
        # 背景を白にして表示
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        axes[i].imshow(img)
        
        # ファイル名から情報抽出
        parts = img_path.stem.split('_')
        if 'lesion' in parts:
            lesion_idx = parts.index('lesion')
            title = f"{parts[0]}_{parts[1]}\n病変{parts[lesion_idx+1]} ({parts[-1]})"
        else:
            title = img_path.stem
            
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # 余った軸を非表示
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'切り抜き病変画像 (最新{n_images}件)', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='切り抜き病変プレビューア')
    parser.add_argument('-d', '--dir', default='extracted_lesions', help='病変画像ディレクトリ')
    parser.add_argument('-n', '--num', type=int, default=12, help='表示する画像数')
    
    args = parser.parse_args()
    
    preview_extracted_lesions(args.dir, args.num)

if __name__ == "__main__":
    main()