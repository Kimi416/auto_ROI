#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCC画像の不正マスクチェック（軽量版）
問題のある画像を特定して、必要に応じて修正
"""

import cv2
import numpy as np
from pathlib import Path
import json

class QuickMaskChecker:
    def __init__(self):
        # OpenCV顔検出器を初期化
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        
        self.incorrect_masks = []
        self.correct_faces = []
        self.correct_trunks = []
        
    def has_black_mask(self, image):
        """画像に黒いマスクがあるかチェック"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray == 0)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_ratio = black_pixels / total_pixels
            return black_ratio > 0.05, black_ratio
        except:
            return False, 0
    
    def detect_face_simple(self, image):
        """シンプルな顔検出"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(50, 50))
            return len(faces) > 0
        except:
            return False
    
    def check_image(self, img_path):
        """画像のマスク状態をチェック"""
        try:
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                return "error"
            
            # 黒いマスクの存在チェック
            has_mask, mask_ratio = self.has_black_mask(image)
            
            # 顔の存在チェック
            has_face = self.detect_face_simple(image)
            
            # 判定
            if has_face and has_mask:
                return "correct_face"  # 顔あり、マスクあり（正常）
            elif has_face and not has_mask:
                return "face_no_mask"  # 顔あり、マスクなし（要マスク）
            elif not has_face and has_mask:
                return "trunk_with_mask"  # 顔なし、マスクあり（要除去）
            else:
                return "correct_trunk"  # 顔なし、マスクなし（正常）
                
        except Exception as e:
            print(f"❌ チェックエラー {img_path.name}: {e}")
            return "error"
    
    def quick_check_all(self):
        """全BCC画像の簡易チェック"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        print(f"🔍 BCC画像マスク状態チェック")
        print(f"📁 対象画像数: {len(bcc_files)}枚")
        print("=" * 40)
        
        status_counts = {
            "correct_face": 0,
            "face_no_mask": 0,
            "trunk_with_mask": 0,
            "correct_trunk": 0,
            "error": 0
        }
        
        problem_files = {
            "face_no_mask": [],
            "trunk_with_mask": []
        }
        
        # チェック実行
        for i, img_path in enumerate(bcc_files, 1):
            if i % 100 == 0:
                print(f"進行状況: {i}/{len(bcc_files)} ...")
            
            status = self.check_image(img_path)
            status_counts[status] += 1
            
            if status in problem_files:
                problem_files[status].append(str(img_path))
        
        # 結果表示
        print(f"\n📊 チェック結果:")
        print(f"  ✅ 正常な顔面画像（マスクあり）: {status_counts['correct_face']}枚")
        print(f"  ✅ 正常な体幹部画像（マスクなし）: {status_counts['correct_trunk']}枚")
        print(f"  ⚠️ 顔面画像でマスクなし: {status_counts['face_no_mask']}枚")
        print(f"  ❌ 体幹部画像に不正マスク: {status_counts['trunk_with_mask']}枚")
        print(f"  ❌ エラー: {status_counts['error']}枚")
        
        # 問題ファイルの詳細
        if problem_files["trunk_with_mask"]:
            print(f"\n❌ 体幹部画像の不正マスク（最初の10個）:")
            for file_path in problem_files["trunk_with_mask"][:10]:
                print(f"  - {Path(file_path).name}")
        
        if problem_files["face_no_mask"]:
            print(f"\n⚠️ 顔面画像でマスクなし（最初の10個）:")
            for file_path in problem_files["face_no_mask"][:10]:
                print(f"  - {Path(file_path).name}")
        
        # 結果を保存
        result = {
            'total_files': len(bcc_files),
            'status_counts': status_counts,
            'problem_files': problem_files
        }
        
        with open('mask_check_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 チェック結果保存: mask_check_result.json")
        
        # 修正が必要かどうかの判定
        total_problems = status_counts['face_no_mask'] + status_counts['trunk_with_mask']
        if total_problems > 0:
            print(f"\n💡 修正が必要な画像: {total_problems}枚")
            print(f"修正ツールの実行を推奨します")
        else:
            print(f"\n🎉 全ての画像が正常な状態です！")
        
        return total_problems == 0

def main():
    print("🔍 BCC画像マスク状態チェック（軽量版）")
    print("=" * 50)
    print("高速で全画像のマスク状態をチェックします")
    
    checker = QuickMaskChecker()
    checker.quick_check_all()

if __name__ == "__main__":
    main()