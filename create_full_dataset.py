#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完全な皮膚病変データセット作成
BCC大幅追加後の全クラス対応
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

def analyze_organized_data():
    """organizedフォルダのデータ分析"""
    print("📊 organizedフォルダのデータ分析")
    
    organized_dir = Path('organized')
    class_counts = {}
    
    for class_dir in organized_dir.iterdir():
        if class_dir.is_dir():
            # 画像ファイルを数える
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                         list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp")) + \
                         list(class_dir.glob("*.tif"))
            
            class_counts[class_dir.name] = len(image_files)
            print(f"  {class_dir.name}: {len(image_files)}枚")
    
    total_images = sum(class_counts.values())
    print(f"\n総画像数: {total_images}枚")
    
    # 不均衡度分析
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"不均衡比: {max_count}:{min_count} = {max_count/min_count:.1f}:1")
    
    return class_counts

def create_full_yolo_dataset():
    """完全なYOLOデータセット作成"""
    print("\n🏗️ 完全なYOLOデータセット作成開始")
    
    # クラスマッピング
    class_mapping = {
        'ADM': 0,
        'Ephelis': 1, 
        'Melasma': 2,
        'Solar lentigo': 3,
        'Nevus': 4,
        'Basal cell carcinoma': 5,
        'Seborrheic keratosis': 6,
        'Malignant melanoma': 7
    }
    
    # 出力ディレクトリ準備
    output_dir = Path('yolo_dataset_full')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # ディレクトリ構造作成
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    organized_dir = Path('organized')
    all_annotations = []
    total_copied = 0
    
    print("\n📁 各クラスの処理:")
    
    for class_name, class_id in class_mapping.items():
        class_dir = organized_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠️ {class_name} フォルダが見つかりません")
            continue
        
        # 画像ファイル取得
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.bmp', '*.tif']:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            print(f"⚠️ {class_name}: 画像ファイルなし")
            continue
        
        print(f"  {class_name}: {len(image_files)}枚 → 処理中...")
        
        # データ分割（train:70%, valid:20%, test:10%）
        random.shuffle(image_files)
        train_count = int(len(image_files) * 0.7)
        valid_count = int(len(image_files) * 0.2)
        
        splits = {
            'train': image_files[:train_count],
            'valid': image_files[train_count:train_count + valid_count],
            'test': image_files[train_count + valid_count:]
        }
        
        for split, files in splits.items():
            for i, img_path in enumerate(files):
                # 画像コピー
                new_name = f"{class_name}_{i:04d}_{split}.jpg"
                dst_img = output_dir / split / 'images' / new_name
                
                try:
                    shutil.copy2(img_path, dst_img)
                    
                    # 仮のラベルファイル作成（後でアノテーション要）
                    label_path = output_dir / split / 'labels' / f"{new_name.replace('.jpg', '.txt')}"
                    with open(label_path, 'w') as f:
                        # 仮の中央位置アノテーション
                        f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
                    
                    # アノテーション記録
                    all_annotations.append({
                        'image_path': str(dst_img),
                        'label_path': str(label_path),
                        'class_name': class_name,
                        'class_id': class_id,
                        'split': split,
                        'original_path': str(img_path),
                        'needs_annotation': True
                    })
                    
                    total_copied += 1
                    
                except Exception as e:
                    print(f"    エラー: {img_path} → {e}")
        
        print(f"    完了: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    
    print(f"\n✅ 総コピー数: {total_copied}枚")
    
    # アノテーション情報保存
    with open('full_dataset_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    # dataset.yaml作成
    yaml_content = f"""path: {output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 8
names: ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis', 'Malignant melanoma']
"""
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    # 統計情報
    print("\n📈 データセット統計:")
    for split in ['train', 'valid', 'test']:
        img_count = len(list((output_dir / split / 'images').glob('*.jpg')))
        print(f"  {split}: {img_count}枚")
    
    return output_dir

def create_annotation_plan():
    """アノテーション計画作成"""
    print("\n📝 アノテーション計画")
    
    with open('full_dataset_annotations.json', 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # クラス別統計
    class_stats = defaultdict(int)
    for ann in annotations:
        class_stats[ann['class_name']] += 1
    
    print("クラス別アノテーション必要数:")
    total_needed = 0
    for class_name, count in sorted(class_stats.items()):
        print(f"  {class_name}: {count}枚")
        total_needed += count
    
    print(f"\n総アノテーション必要数: {total_needed}枚")
    print("\n💡 推奨アノテーション手順:")
    print("1. 少数クラスから開始（Basal cell carcinoma等）")
    print("2. 各クラス50-100枚程度を先にアノテーション")
    print("3. 初期学習で検証")
    print("4. 段階的に全データをアノテーション")
    
    # 優先度付きアノテーション計画
    priority_plan = {
        'Phase 1 (高優先度)': ['Basal cell carcinoma', 'Malignant melanoma'],
        'Phase 2 (中優先度)': ['ADM', 'Solar lentigo', 'Seborrheic keratosis'],
        'Phase 3 (標準)': ['Ephelis', 'Nevus', 'Melasma']
    }
    
    print("\n🎯 段階的アノテーション計画:")
    for phase, classes in priority_plan.items():
        print(f"\n{phase}:")
        phase_total = sum(class_stats[cls] for cls in classes if cls in class_stats)
        for cls in classes:
            if cls in class_stats:
                print(f"  - {cls}: {class_stats[cls]}枚")
        print(f"  小計: {phase_total}枚")

def main():
    print("🔬 完全な皮膚病変データセット作成")
    print("=" * 60)
    print("BCC大幅追加後の全データ対応")
    
    # 1. データ分析
    class_counts = analyze_organized_data()
    
    # 2. YOLOデータセット作成
    dataset_dir = create_full_yolo_dataset()
    
    # 3. アノテーション計画
    create_annotation_plan()
    
    print(f"\n🎉 データセット作成完了!")
    print(f"出力先: {dataset_dir}")
    print("\n次のステップ:")
    print("1. python yolo_annotator.py でアノテーション開始")
    print("2. 段階的にアノテーションを実行")
    print("3. 適切なタイミングで学習開始")

if __name__ == "__main__":
    random.seed(42)  # 再現性のため
    main()