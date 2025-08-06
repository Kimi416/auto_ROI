#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC画像の破損ファイルをチェック
"""

import cv2
from pathlib import Path
import json

def check_bcc_images():
    """BCC画像の破損チェック"""
    bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
    
    if not bcc_dir.exists():
        print(f"❌ BCCフォルダが見つかりません: {bcc_dir}")
        return
    
    # BCC画像ファイルを取得
    bcc_files = list(bcc_dir.glob("*.jpg")) + list(bcc_dir.glob("*.JPG"))
    
    print(f"🔍 BCC画像の破損チェック開始")
    print(f"📁 総ファイル数: {len(bcc_files)}枚")
    print("=" * 50)
    
    corrupt_files = []
    valid_files = []
    
    for i, img_path in enumerate(bcc_files):
        print(f"[{i+1}/{len(bcc_files)}] {img_path.name}", end=" ... ")
        
        try:
            # 画像読み込みテスト
            image = cv2.imread(str(img_path))
            if image is None:
                print("❌ 読み込み失敗")
                corrupt_files.append(str(img_path))
            else:
                # 基本的な画像情報チェック
                h, w, c = image.shape
                if h < 10 or w < 10:
                    print(f"❌ サイズ異常 ({w}x{h})")
                    corrupt_files.append(str(img_path))
                else:
                    print("✅ OK")
                    valid_files.append(str(img_path))
        
        except Exception as e:
            print(f"❌ エラー: {e}")
            corrupt_files.append(str(img_path))
        
        # 50ファイルごとに進行状況表示
        if (i + 1) % 50 == 0:
            print(f"\n--- 中間結果 ({i+1}枚処理) ---")
            print(f"正常: {len(valid_files)}枚")
            print(f"破損: {len(corrupt_files)}枚")
            print("-" * 30)
    
    # 結果保存
    result = {
        'total_files': len(bcc_files),
        'valid_files': len(valid_files),
        'corrupt_files': len(corrupt_files),
        'corrupt_file_list': corrupt_files
    }
    
    with open('bcc_corrupt_check.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 50)
    print(f"📊 BCC画像破損チェック結果:")
    print(f"  総ファイル数: {len(bcc_files)}枚")
    print(f"  正常ファイル: {len(valid_files)}枚")
    print(f"  破損ファイル: {len(corrupt_files)}枚")
    print(f"💾 結果保存: bcc_corrupt_check.json")
    
    if corrupt_files:
        print(f"\n⚠️ 破損ファイル（最初の10個）:")
        for corrupt_file in corrupt_files[:10]:
            print(f"  - {Path(corrupt_file).name}")
        
        print(f"\n💡 対処方法:")
        print(f"1. 破損ファイルを organized から再コピー")
        print(f"2. 破損ファイルを削除して処理続行")
    
    return len(corrupt_files) == 0

if __name__ == "__main__":
    check_bcc_images()