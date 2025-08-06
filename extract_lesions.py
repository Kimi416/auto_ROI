#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
アノテーション済みデータからYOLO学習用データセット生成
- localStorageからアノテーションデータを取得
- 病変領域を切り出して個別画像として保存
- YOLO形式のラベルファイル生成
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import sqlite3
import base64

def create_yolo_dataset_structure(output_dir):
    """YOLO学習用ディレクトリ構造を作成"""
    dataset_dir = Path(output_dir)
    
    # ディレクトリ作成
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "cropped_lesions").mkdir(parents=True, exist_ok=True)
    
    # dataset.yaml作成
    yaml_content = """train: images/train
val: images/val
nc: 8
names: ['Melasma', 'Solar_lentigo', 'Nevus', 'ADM', 'Ephelis', 'Seborrheic_keratosis', 'Basal_cell_carcinoma', 'Malignant_melanoma']
"""
    
    with open(dataset_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    return dataset_dir

def extract_browser_annotations(browser_data_dir=None):
    """ブラウザのlocalStorageからアノテーションデータを抽出"""
    annotations = {}
    
    # Chrome/Edge用のlocalStorageパス（macOS）
    possible_paths = [
        os.path.expanduser("~/Library/Application Support/Google Chrome/Default/Local Storage/leveldb"),
        os.path.expanduser("~/Library/Application Support/Microsoft Edge/Default/Local Storage/leveldb"),
        os.path.expanduser("~/Library/Application Support/Chromium/Default/Local Storage/leveldb")
    ]
    
    if browser_data_dir:
        possible_paths.insert(0, browser_data_dir)
    
    print("ブラウザのlocalStorageからアノテーションデータを検索中...")
    
    # 手動でJSONファイルを読み込む方法も提供
    manual_export_path = Path("exported_annotations.json")
    if manual_export_path.exists():
        print(f"手動エクスポートファイルを発見: {manual_export_path}")
        with open(manual_export_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        return annotations
    
    print("""
ブラウザのlocalStorageから直接読み取れませんでした。
手動でデータをエクスポートしてください：

1. アノテーションツールを開く
2. 開発者ツール（F12）を開く  
3. Consoleタブで以下を実行：

const data = {};
Object.keys(localStorage).filter(k => k.startsWith('annotation_')).forEach(key => {
    data[key] = JSON.parse(localStorage.getItem(key));
});
console.log(JSON.stringify(data, null, 2));

4. 出力されたJSONを 'exported_annotations.json' ファイルに保存
""")
    
    return {}

def path_to_bbox(path_points, img_width, img_height):
    """描画パスからバウンディングボックスを計算"""
    if len(path_points) < 3:
        return None
    
    xs = [p['x'] for p in path_points]
    ys = [p['y'] for p in path_points]
    
    min_x = max(0, min(xs))
    max_x = min(img_width, max(xs))
    min_y = max(0, min(ys))
    max_y = min(img_height, max(ys))
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width < 10 or height < 10:  # 小さすぎる領域は無視
        return None
    
    # YOLO形式（中心座標、正規化）
    center_x = (min_x + max_x) / 2 / img_width
    center_y = (min_y + max_y) / 2 / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return {
        'bbox': [center_x, center_y, norm_width, norm_height],
        'pixel_bbox': [int(min_x), int(min_y), int(width), int(height)]
    }

def create_lesion_mask(path_points, img_width, img_height):
    """描画パスからマスクを作成"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if len(path_points) < 3:
        return mask
    
    # パスをOpenCVのポリゴンに変換
    points = np.array([[int(p['x']), int(p['y'])] for p in path_points], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    return mask

def main():
    parser = argparse.ArgumentParser(description='アノテーションからYOLO学習用データセット生成')
    parser.add_argument('images_dir', help='マスク済み画像ディレクトリ')
    parser.add_argument('-o', '--output', default='yolo_dataset', help='出力ディレクトリ')
    parser.add_argument('--val-split', type=float, default=0.2, help='検証用データの割合')
    parser.add_argument('--crop-lesions', action='store_true', help='病変領域を個別画像として切り出し')
    
    args = parser.parse_args()
    
    # データセット構造作成
    dataset_dir = create_yolo_dataset_structure(args.output)
    print(f"データセット構造を作成: {dataset_dir}")
    
    # アノテーションデータ抽出
    annotations = extract_browser_annotations()
    
    if not annotations:
        print("アノテーションデータが見つかりません。手動エクスポートが必要です。")
        return
    
    print(f"アノテーションファイル数: {len(annotations)}")
    
    # 病変タイプマッピング
    disease_mapping = {
        'Melasma': 0,
        'Solar_lentigo': 1, 'Solar lentigo': 1,
        'Nevus': 2,
        'ADM': 3,
        'Ephelis': 4,
        'Seborrheic_keratosis': 5, 'Seborrheic keratosis': 5,
        'Basal_cell_carcinoma': 6, 'Basal cell carcinoma': 6,
        'Malignant_melanoma': 7, 'Malignant melanoma': 7
    }
    
    images_dir = Path(args.images_dir)
    processed_count = 0
    lesion_count = 0
    cropped_count = 0
    
    # train/val分割用
    file_list = list(annotations.keys())
    val_count = int(len(file_list) * args.val_split)
    val_files = set(file_list[:val_count])
    
    for ann_key, ann_data in tqdm(annotations.items(), desc="データセット生成中"):
        filename = ann_data['fileName']
        
        # 画像ファイルを検索
        image_path = None
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']:
            potential_path = images_dir / filename.replace(Path(filename).suffix, f'.{ext}')
            if potential_path.exists():
                image_path = potential_path
                break
        
        # organized階層も検索
        if not image_path:
            for subdir in images_dir.rglob("*"):
                if subdir.is_dir():
                    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']:
                        potential_path = subdir / filename.replace(Path(filename).suffix, f'.{ext}')
                        if potential_path.exists():
                            image_path = potential_path
                            break
                    if image_path:
                        break
        
        if not image_path:
            print(f"画像が見つかりません: {filename}")
            continue
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # train/val分割
        split = "val" if ann_key in val_files else "train"
        
        # 出力パス
        base_name = Path(filename).stem
        output_image_path = dataset_dir / "images" / split / f"{base_name}.jpg"
        output_label_path = dataset_dir / "labels" / split / f"{base_name}.txt"
        
        # 画像コピー
        cv2.imwrite(str(output_image_path), image)
        
        # ラベルファイル生成
        yolo_lines = []
        
        for i, annotation in enumerate(ann_data['annotations']):
            disease = annotation['disease']
            class_id = disease_mapping.get(disease, 0)
            
            # バウンディングボックス計算
            bbox_data = path_to_bbox(annotation['path'], img_width, img_height)
            if not bbox_data:
                continue
            
            bbox = bbox_data['bbox']
            pixel_bbox = bbox_data['pixel_bbox']
            
            # YOLOラベル行
            yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
            lesion_count += 1
            
            # 病変領域を切り出し
            if args.crop_lesions:
                x, y, w, h = pixel_bbox
                
                # マージン追加
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img_width, x + w + margin)
                y2 = min(img_height, y + h + margin)
                
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    crop_name = f"{base_name}_lesion_{i}_{disease}.jpg"
                    crop_path = dataset_dir / "cropped_lesions" / crop_name
                    cv2.imwrite(str(crop_path), cropped)
                    cropped_count += 1
        
        # ラベルファイル保存
        if yolo_lines:
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        
        processed_count += 1
    
    print(f"\n=== データセット生成完了 ===")
    print(f"処理済み画像: {processed_count}枚")
    print(f"検出病変数: {lesion_count}個")
    if args.crop_lesions:
        print(f"切り出し病変: {cropped_count}個")
    print(f"出力先: {dataset_dir}")
    print(f"dataset.yaml: 学習時にこのファイルを指定してください")

if __name__ == '__main__':
    main()