#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
顔面画像の規定通りマスク処理
顔が検出された画像に確実に最適なT字マスクを適用
"""

import cv2
import numpy as np
from pathlib import Path
import json

class FaceMaskFixer:
    def __init__(self):
        # OpenCV顔検出器を初期化（複数の検出器を使用）
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        
        self.fixed_faces = []
        self.already_masked = []
        self.no_face_detected = []
        
    def detect_faces_comprehensive(self, image):
        """包括的な顔検出（正面顔・横顔・目検出の組み合わせ）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = []
            
            # 1. 正面顔検出（複数のパラメータで試行）
            for scale_factor in [1.1, 1.2, 1.3]:
                for min_neighbors in [3, 4, 5]:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=scale_factor, 
                        minNeighbors=min_neighbors,
                        minSize=(40, 40),
                        maxSize=(300, 300)
                    )
                    for face in faces:
                        detected_faces.append(face)
            
            # 2. 横顔検出
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
                maxSize=(300, 300)
            )
            for profile in profiles:
                detected_faces.append(profile)
            
            # 重複を除去（重複する領域を統合）
            if not detected_faces:
                return None
            
            # 最大の顔を選択
            best_face = max(detected_faces, key=lambda x: x[2] * x[3])
            x, y, w, h = best_face
            
            # 目検出で顔の妥当性を確認
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 2)
            
            # 目が検出されるか、または顔が十分大きい場合は有効な顔とする
            if len(eyes) > 0 or (w > 60 and h > 60):
                return (x, y, w, h)
            
            return None
            
        except Exception as e:
            print(f"❌ 顔検出エラー: {e}")
            return None
    
    def apply_regulated_t_mask(self, image, face_region):
        """規定通りの最適T字マスク適用"""
        try:
            x, y, w, h = face_region
            face_center_x = x + w // 2
            
            # organized_advanced_maskedスタイルの最適T字マスク
            # 眉毛から目の領域（横棒）- より確実にカバー
            eyebrow_top = max(0, y - int(h * 0.3))  # 眉毛の上部（十分な余裕）
            eye_bottom = y + int(h * 0.5)  # 目の下部（十分な余裕）
            horizontal_left = x + int(w * 0.02)  # 顔幅のほぼ全体
            horizontal_right = x + w - int(w * 0.02)  # 顔幅のほぼ全体
            
            # 鼻の領域（縦棒上部）
            nose_top = y + int(h * 0.25)
            nose_bottom = y + int(h * 0.75)
            nose_left = face_center_x - int(w * 0.15)
            nose_right = face_center_x + int(w * 0.15)
            
            # 口の領域（縦棒下部）
            mouth_top = y + int(h * 0.55)
            mouth_bottom = y + h - int(h * 0.05)
            mouth_left = face_center_x - int(w * 0.25)
            mouth_right = face_center_x + int(w * 0.25)
            
            # マスクを適用
            masked = image.copy()
            
            # 横棒（眉毛から目）を黒く塗りつぶし
            cv2.rectangle(masked, (horizontal_left, eyebrow_top), 
                         (horizontal_right, eye_bottom), (0, 0, 0), -1)
            
            # 鼻の領域を黒く塗りつぶし
            cv2.rectangle(masked, (nose_left, nose_top), 
                         (nose_right, nose_bottom), (0, 0, 0), -1)
            
            # 口の領域を黒く塗りつぶし
            cv2.rectangle(masked, (mouth_left, mouth_top), 
                         (mouth_right, mouth_bottom), (0, 0, 0), -1)
            
            return masked
            
        except Exception as e:
            print(f"❌ マスク適用エラー: {e}")
            return image
    
    def has_sufficient_mask(self, image):
        """十分なマスクがあるかチェック"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray == 0)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_ratio = black_pixels / total_pixels
            return black_ratio > 0.15  # 15%以上が黒ければ十分マスクされている
        except:
            return False
    
    def fix_face_mask(self, img_path):
        """顔面画像のマスクを修正"""
        try:
            print(f"📸 処理中: {img_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ 画像読み込み失敗")
                return False
            
            # 包括的な顔検出
            face_region = self.detect_faces_comprehensive(image)
            
            if face_region is None:
                print(f"🫁 顔検出なし（体幹部画像と判定）")
                self.no_face_detected.append(str(img_path))
                return True
            
            # 現在のマスク状況をチェック
            has_mask = self.has_sufficient_mask(image)
            
            if has_mask:
                print(f"👤 既にマスク済み（スキップ）")
                self.already_masked.append(str(img_path))
                return True
            
            # 顔面画像で十分なマスクがない場合 → 規定通りマスクを適用
            print(f"👤 顔面画像: 規定通りマスク適用")
            masked_image = self.apply_regulated_t_mask(image, face_region)
            
            # 保存
            cv2.imwrite(str(img_path), masked_image)
            self.fixed_faces.append(str(img_path))
            print(f"✅ 規定通りT字マスク適用完了")
            
            return True
            
        except Exception as e:
            print(f"❌ 処理エラー: {e}")
            return False
    
    def fix_all_face_masks(self):
        """全BCC画像の顔面マスクを修正"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        print(f"👤 BCC顔面画像の規定通りマスク処理")
        print(f"📁 対象画像数: {len(bcc_files)}枚")
        print("=" * 50)
        
        for i, img_path in enumerate(bcc_files, 1):
            print(f"[{i}/{len(bcc_files)}] ", end="")
            self.fix_face_mask(img_path)
            
            # 100枚ごとに進行状況を表示
            if i % 100 == 0:
                print(f"\n--- 進行状況 ({i}枚処理) ---")
                print(f"マスク修正: {len(self.fixed_faces)}枚")
                print(f"既にマスク済み: {len(self.already_masked)}枚")
                print(f"顔検出なし: {len(self.no_face_detected)}枚")
                print("-" * 30)
        
        # 結果の保存
        result = {
            'total_files': len(bcc_files),
            'fixed_faces': len(self.fixed_faces),
            'already_masked': len(self.already_masked),
            'no_face_detected': len(self.no_face_detected),
            'fixed_face_list': self.fixed_faces,
            'already_masked_list': self.already_masked,
            'no_face_list': self.no_face_detected
        }
        
        with open('face_mask_fix_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 50)
        print(f"📊 顔面マスク修正結果:")
        print(f"  総ファイル数: {len(bcc_files)}枚")
        print(f"  マスク修正: {len(self.fixed_faces)}枚")
        print(f"  既にマスク済み: {len(self.already_masked)}枚")
        print(f"  顔検出なし（体幹部）: {len(self.no_face_detected)}枚")
        print(f"💾 結果保存: face_mask_fix_result.json")
        
        return True

def main():
    print("👤 BCC顔面画像規定通りマスク処理")
    print("=" * 50)
    print("📌 顔が検出された画像に確実に最適なT字マスクを適用")
    print("🎭 organized_advanced_maskedスタイルの規定通りマスク")
    
    fixer = FaceMaskFixer()
    
    print(f"\n処理を開始しますか？ (Enter=開始, Ctrl+C=中止)")
    try:
        input()
    except KeyboardInterrupt:
        print("\n中止されました")
        return
    
    if fixer.fix_all_face_masks():
        print(f"\n🎉 顔面マスク修正完了!")
        print(f"\n💡 次のステップ:")
        print(f"1. サンプルチェックで結果確認")
        print(f"2. アノテーション作業の開始")
    else:
        print(f"\n❌ 処理でエラーが発生しました")

if __name__ == "__main__":
    main()