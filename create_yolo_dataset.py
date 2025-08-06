#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOアノテーション結果からYOLO学習用データセットを作成
"""

import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def create_yolo_dataset():
    # アノテーションデータを読み込み
    with open('yolo_annotations.json', 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"📊 総アノテーション数: {len(annotations)}")
    
    # データセットディレクトリを作成
    dataset_dir = Path('yolo_dataset')
    
    for split in ['train', 'valid', 'test']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # データを分割（70%学習、20%検証、10%テスト）
    random.shuffle(annotations)
    total = len(annotations)
    train_size = int(total * 0.7)
    valid_size = int(total * 0.2)
    
    train_data = annotations[:train_size]
    valid_data = annotations[train_size:train_size + valid_size]
    test_data = annotations[train_size + valid_size:]
    
    print(f"📈 データ分割:")
    print(f"  学習用: {len(train_data)}枚")
    print(f"  検証用: {len(valid_data)}枚") 
    print(f"  テスト用: {len(test_data)}枚")
    
    # 各データセットを作成
    for split, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        print(f"\n🔄 {split}データセット作成中...")
        
        for item in tqdm(data):
            # 画像をコピー
            src_image = Path(item['image_path'])
            dst_image = dataset_dir / split / 'images' / src_image.name
            shutil.copy2(src_image, dst_image)
            
            # ラベルファイルを作成
            label_file = dataset_dir / split / 'labels' / (src_image.stem + '.txt')
            
            with open(label_file, 'w') as f:
                for ann in item['annotations']:
                    class_id = ann['class_id']
                    x_center = ann['x_center']
                    y_center = ann['y_center']
                    width = ann['width']
                    height = ann['height']
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # dataset.yamlファイルを作成
    yaml_content = f"""# 皮膚病変検出データセット
path: {dataset_dir.absolute()}
train: train/images
val: valid/images
test: test/images

# クラス
nc: 8
names:
  0: ADM
  1: Ephelis
  2: Melasma
  3: Solar_lentigo
  4: Nevus
  5: Basal_cell_carcinoma
  6: Seborrheic_keratosis
  7: Malignant_melanoma
"""
    
    with open(dataset_dir / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✅ YOLOデータセット作成完了: {dataset_dir}")
    print(f"📁 dataset.yaml: {dataset_dir / 'dataset.yaml'}")
    
    return dataset_dir

if __name__ == "__main__":
    create_yolo_dataset()