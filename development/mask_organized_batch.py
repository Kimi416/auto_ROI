#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
organizedフォルダ全体をバッチでマスク処理するスクリプト
進行状況を保存し、中断しても再開可能
"""

import json
from pathlib import Path
import sys
import time

# mask.pyをインポート
sys.path.append(str(Path(__file__).parent))
from mask import FaceTMasker

def batch_mask_organized():
    """organizedフォルダをバッチでマスク処理"""
    
    input_dir = Path("organized")
    output_dir = Path("organized_masked_improved")
    progress_file = Path("mask_progress_improved.json")
    
    # 進行状況を読み込みまたは初期化
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"既存の進行状況を読み込みました: {len(progress['completed'])}枚完了")
    else:
        progress = {
            "completed": [],
            "failed": [],
            "start_time": time.time()
        }
    
    # マスカーを初期化
    masker = FaceTMasker()
    
    # 画像ファイルを収集
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP', '*.tif', '*.TIF']:
        image_files.extend(input_dir.rglob(ext))
    
    print(f"処理対象: {len(image_files)}枚の画像")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 処理開始
    for i, image_path in enumerate(image_files):
        # 既に処理済みの場合はスキップ
        if str(image_path) in progress['completed'] or str(image_path) in progress['failed']:
            continue
        
        # 相対パスを保持してサブディレクトリ構造を維持
        relative_path = image_path.relative_to(input_dir)
        output_file = output_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 進行状況を表示
        completed_count = len(progress['completed'])
        failed_count = len(progress['failed'])
        total_processed = completed_count + failed_count
        
        print(f"\n[{total_processed + 1}/{len(image_files)}] 処理中: {relative_path}")
        
        # マスク処理を実行
        try:
            success = masker.mask_face_t_shape(image_path, output_file)
            
            if success:
                progress['completed'].append(str(image_path))
                print(f"  ✅ 成功")
            else:
                progress['failed'].append(str(image_path))
                print(f"  ❌ 顔検出失敗")
        
        except Exception as e:
            progress['failed'].append(str(image_path))
            print(f"  ❌ エラー: {e}")
        
        # 10枚ごとに進行状況を保存
        if (total_processed + 1) % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"  📁 進行状況を保存しました")
    
    # 最終結果を保存
    progress['end_time'] = time.time()
    progress['total_time'] = progress['end_time'] - progress['start_time']
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # 失敗したファイルのリストを別途保存
    if progress['failed']:
        failed_list_path = output_dir / 'failed_files.json'
        with open(failed_list_path, 'w') as f:
            json.dump(progress['failed'], f, indent=2)
    
    # 統計を表示
    print("\n" + "="*60)
    print("処理完了!")
    print("="*60)
    print(f"成功: {len(progress['completed'])}枚")
    print(f"失敗: {len(progress['failed'])}枚")
    print(f"成功率: {len(progress['completed']) / len(image_files) * 100:.1f}%")
    print(f"処理時間: {progress['total_time'] / 60:.1f}分")
    print(f"\n出力先: {output_dir}")
    
    # カテゴリ別の統計
    category_stats = {}
    for path in progress['completed']:
        category = Path(path).parts[-2]
        category_stats[category] = category_stats.get(category, 0) + 1
    
    print("\nカテゴリ別成功数:")
    for category, count in sorted(category_stats.items()):
        print(f"  {category}: {count}枚")

if __name__ == "__main__":
    print("organizedフォルダのマスク処理を開始します")
    print("中断してもmask_progress_improved.jsonから再開可能です")
    print("Ctrl+Cで安全に中断できます")
    
    try:
        batch_mask_organized()
    except KeyboardInterrupt:
        print("\n\n処理を中断しました。次回実行時に自動的に再開されます。")