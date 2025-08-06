#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8を使用した皮膚病変検出のセットアップスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path

def install_yolo():
    """YOLOv8と必要なパッケージをインストール"""
    print("YOLOv8環境をセットアップしています...")
    
    packages = [
        "ultralytics",  # YOLOv8
        "torch",        # PyTorch
        "torchvision",  # PyTorch Vision
        "torchaudio",   # PyTorch Audio (依存関係)
        "pandas",       # データ処理
        "seaborn",      # 可視化
        "matplotlib",   # グラフ描画
        "pillow",       # 画像処理
        "pyyaml",       # YAML設定ファイル
        "tensorboard",  # 学習の可視化
        "labelImg",     # アノテーションツール（オプション）
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ YOLOv8環境のセットアップが完了しました！")

def create_dataset_structure():
    """YOLO学習用のデータセット構造を作成"""
    base_path = Path("/Users/iinuma/Desktop/自動ROI/yolo_dataset")
    
    # ディレクトリ構造を作成
    dirs = [
        "train/images",
        "train/labels",
        "valid/images", 
        "valid/labels",
        "test/images",
        "test/labels"
    ]
    
    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ データセット構造を作成しました: {base_path}")
    
    # データセット設定ファイルを作成
    yaml_content = """# YOLOv8 Dataset Configuration
# 皮膚病変検出用の設定

# Dataset paths
path: /Users/iinuma/Desktop/自動ROI/yolo_dataset  # dataset root dir
train: train/images  # train images 
val: valid/images    # val images
test: test/images    # test images (optional)

# Classes
names:
  0: ADM
  1: Ephelis
  2: Melasma
  3: Solar_lentigo
  4: Nevus
  5: Basal_cell_carcinoma
  6: Seborrheic_keratosis
  7: Malignant_melanoma

# Number of classes
nc: 8
"""
    
    yaml_path = base_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ データセット設定ファイルを作成しました: {yaml_path}")
    
    return base_path

def main():
    print("=== YOLOv8 皮膚病変検出システムのセットアップ ===\n")
    
    # YOLOをインストール
    install_yolo()
    
    # データセット構造を作成
    dataset_path = create_dataset_structure()
    
    print("\n=== セットアップ完了 ===")
    print("\n次のステップ:")
    print("1. マスク済み画像から病変部分をアノテーション")
    print("2. アノテーションツール: LabelImg または Roboflow")
    print("3. YOLOフォーマット（.txt）でラベルを保存")
    print(f"4. 画像とラベルを {dataset_path} に配置")
    print("5. train_yolo.py で学習を開始")

if __name__ == "__main__":
    main()