#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
マスク済みBCC画像を organized_advanced_masked フォルダに統合
"""

import shutil
import json
from pathlib import Path

def integrate_masked_bcc():
    """マスク済みBCC画像を統合"""
    print("🔗 マスク済みBCC画像の統合開始")
    print("=" * 50)
    
    # ソースとターゲットの設定
    source_dir = Path('yolo_dataset')
    target_dir = Path('organized_advanced_masked/Basal cell carcinoma')
    
    # ターゲットディレクトリの作成
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # BCC追加情報を読み込み
    with open('bcc_additions.json', 'r', encoding='utf-8') as f:
        bcc_files = json.load(f)
    
    # マスク処理結果を読み込み
    mask_result_file = Path('bcc_face_mask_result.json')
    if mask_result_file.exists():
        with open(mask_result_file, 'r', encoding='utf-8') as f:
            mask_result = json.load(f)
        face_files = {info['image_path'] for info in mask_result.get('face_files', [])}
        print(f"📋 マスク処理結果: {len(face_files)}枚の顔面画像")
    else:
        face_files = set()
        print("⚠️ マスク処理結果ファイルが見つかりません - 全画像を対象とします")
    
    # 既存のBCCファイル数を確認
    existing_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.JPG"))
    print(f"📁 既存のBCC画像数: {len(existing_files)}枚")
    
    # BCC画像を統合
    copied_count = 0
    face_count = 0
    trunk_count = 0
    
    for file_info in bcc_files:
        source_path = Path(file_info['image_path'])
        
        if not source_path.exists():
            print(f"❌ ソースファイルが見つかりません: {source_path}")
            continue
        
        # 元のファイル名を取得（organized フォルダの命名規則に合わせる）
        original_path = file_info['original_path']
        original_name = Path(original_path).name
        
        # ターゲットファイル名を決定
        target_path = target_dir / original_name
        
        # 重複チェック
        if target_path.exists():
            print(f"⏭️ 既存ファイルをスキップ: {original_name}")
            continue
        
        try:
            # ファイルをコピー
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
            # 顔面/体幹部の判定
            if str(source_path) in face_files:
                face_count += 1
                status = "顔面(マスク済み)"
            else:
                trunk_count += 1
                status = "体幹部"
            
            print(f"✅ {status}: {original_name}")
            
        except Exception as e:
            print(f"❌ コピーエラー: {original_name} → {e}")
    
    # 統合後の確認
    final_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.JPG"))
    
    print(f"\n📊 統合結果:")
    print(f"  新規追加: {copied_count}枚")
    print(f"    - 顔面(マスク済み): {face_count}枚")
    print(f"    - 体幹部: {trunk_count}枚")
    print(f"  最終合計: {len(final_files)}枚")
    
    # 統合情報を保存
    integration_info = {
        'source_directory': str(source_dir),
        'target_directory': str(target_dir),
        'existing_files': len(existing_files),
        'new_files_added': copied_count,
        'face_images_masked': face_count,
        'trunk_images': trunk_count,
        'total_files': len(final_files),
        'integration_date': '2025-08-05'
    }
    
    with open('bcc_integration_result.json', 'w', encoding='utf-8') as f:
        json.dump(integration_info, f, ensure_ascii=False, indent=2)
    
    print(f"💾 統合情報保存: bcc_integration_result.json")
    
    return copied_count, len(final_files)

def main():
    print("🔗 BCC画像統合ツール")
    print("マスク済みBCC → organized_advanced_masked/Basal cell carcinoma")
    
    copied, total = integrate_masked_bcc()
    
    if copied > 0:
        print(f"\n🎉 統合完了!")
        print(f"📈 BCC画像数: {total}枚 (新規: {copied}枚)")
        print(f"📂 統合先: organized_advanced_masked/Basal cell carcinoma/")
        
        print(f"\n💡 次のステップ:")
        print(f"1. 統合されたBCC画像でアノテーション作業")
        print(f"2. organized_advanced_masked フォルダを使用して学習")
    else:
        print(f"\n⚠️ 新規追加ファイルなし")
        print(f"📊 現在のBCC画像数: {total}枚")

if __name__ == "__main__":
    main()