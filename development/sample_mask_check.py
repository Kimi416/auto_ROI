#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC画像のサンプルマスクチェック
最初の20枚をチェックして状況を確認
"""

import cv2
import numpy as np
from pathlib import Path

def check_sample_images():
    """サンプル画像をチェック"""
    bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
    
    if not bcc_dir.exists():
        print(f"❌ BCCフォルダが見つかりません")
        return
    
    # 最初の20枚を取得
    bcc_files = list(bcc_dir.glob("*.jpg"))[:20] + list(bcc_dir.glob("*.JPG"))[:20]
    bcc_files = bcc_files[:20]  # 20枚に制限
    
    print(f"🔍 BCC画像サンプルチェック")
    print(f"📁 チェック対象: {len(bcc_files)}枚")
    print("=" * 50)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    results = []
    
    for i, img_path in enumerate(bcc_files, 1):
        try:
            print(f"[{i}/{len(bcc_files)}] {img_path.name}", end=" ... ")
            
            # 画像読み込み（エラーハンドリング強化）
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                print("❌ 読み込み失敗")
                results.append({"file": img_path.name, "status": "error"})
                continue
            
            # 黒いマスクの存在チェック
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray == 0)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_ratio = black_pixels / total_pixels
            has_mask = black_ratio > 0.05
            
            # 顔検出
            faces = face_cascade.detectMultiScale(gray, 1.3, 3, minSize=(30, 30))
            has_face = len(faces) > 0
            
            # 判定
            if has_face and has_mask:
                status = "👤 顔面(マスクあり)"
                category = "correct_face"
            elif has_face and not has_mask:
                status = "⚠️ 顔面(マスクなし)"
                category = "face_no_mask"
            elif not has_face and has_mask:
                status = "❌ 体幹部(不正マスク)"
                category = "trunk_with_mask"
            else:
                status = "🫁 体幹部(正常)"
                category = "correct_trunk"
            
            print(f"{status} (黒領域: {black_ratio*100:.1f}%)")
            
            results.append({
                "file": img_path.name,
                "status": category,
                "has_face": has_face,
                "has_mask": has_mask,
                "black_ratio": black_ratio
            })
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            results.append({"file": img_path.name, "status": "error"})
    
    # 結果のサマリー
    print(f"\n📊 サンプル結果:")
    categories = {}
    for result in results:
        status = result["status"]
        categories[status] = categories.get(status, 0) + 1
    
    for status, count in categories.items():
        print(f"  {status}: {count}枚")
    
    # 問題のある画像を特定
    problems = [r for r in results if r["status"] in ["face_no_mask", "trunk_with_mask"]]
    if problems:
        print(f"\n⚠️ 問題のある画像:")
        for problem in problems:
            print(f"  - {problem['file']}: {problem['status']}")
    else:
        print(f"\n✅ サンプル画像は全て正常です")

if __name__ == "__main__":
    check_sample_images()