#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
不正なマスクの修正
顔がない画像のマスクを除去し、顔がある画像のみに適切なマスクを適用
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil

class MaskFixer:
    def __init__(self):
        # OpenCV顔検出器を初期化（より厳密な設定）
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        self.backup_dir = Path('organized_bcc_backup')
        
        self.fixed_count = 0
        self.face_images = []
        self.trunk_images = []
        
    def create_backup(self):
        """バックアップを作成"""
        if self.backup_dir.exists():
            print(f"📁 既存のバックアップフォルダを使用: {self.backup_dir}")
            return True
        
        try:
            self.backup_dir.mkdir(exist_ok=True)
            print(f"📁 バックアップフォルダ作成: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ バックアップフォルダ作成失敗: {e}")
            return False
    
    def is_face_image_strict(self, image):
        """厳密な顔検出（複数の方法で確認）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 正面顔検出（厳密な設定）
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,  # より厳密に
                minSize=(60, 60),  # 最小サイズを大きく
                maxSize=(400, 400)  # 最大サイズを制限
            )
            
            if len(faces) > 0:
                # 検出された顔が妥当なサイズか確認
                for (x, y, w, h) in faces:
                    if w > 50 and h > 50:  # 適度なサイズの顔
                        # 目の検出でさらに確認
                        face_roi = gray[y:y+h, x:x+w]
                        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                        if len(eyes) >= 1:  # 最低1つの目が検出されれば顔と判定
                            return True, (x, y, w, h)
            
            # 2. 横顔検出
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                maxSize=(400, 400)
            )
            
            if len(profiles) > 0:
                for (x, y, w, h) in profiles:
                    if w > 50 and h > 50:
                        return True, (x, y, w, h)
            
            return False, None
            
        except Exception as e:
            print(f"❌ 顔検出エラー: {e}")
            return False, None
    
    def has_black_regions(self, image):
        """画像に不自然な黒い領域（マスク）があるかチェック"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 完全な黒領域を検出
            black_mask = (gray == 0)
            black_pixels = np.sum(black_mask)
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # 画像の5%以上が黒ピクセルの場合、マスクが適用されている可能性
            black_ratio = black_pixels / total_pixels
            
            return black_ratio > 0.05, black_ratio
            
        except Exception as e:
            print(f"❌ 黒領域検出エラー: {e}")
            return False, 0
    
    def apply_optimal_mask(self, image, face_region):
        """最適なT字マスクを適用"""
        try:
            x, y, w, h = face_region
            face_center_x = x + w // 2
            
            # T字マスクの領域を計算（organized_advanced_maskedスタイル）
            # 眉毛から目の領域（横棒）
            eyebrow_top = max(0, y - int(h * 0.25))
            eye_bottom = y + int(h * 0.45)
            horizontal_left = x + int(w * 0.05)
            horizontal_right = x + w - int(w * 0.05)
            
            # 鼻の領域
            nose_top = y + int(h * 0.3)
            nose_bottom = y + int(h * 0.7)
            nose_left = face_center_x - int(w * 0.12)
            nose_right = face_center_x + int(w * 0.12)
            
            # 口の領域
            mouth_top = y + int(h * 0.6)
            mouth_bottom = y + h - int(h * 0.1)
            mouth_left = face_center_x - int(w * 0.18)
            mouth_right = face_center_x + int(w * 0.18)
            
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
    
    def fix_image_mask(self, img_path):
        """画像のマスクを修正"""
        try:
            print(f"📸 処理中: {img_path.name}")
            
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ 画像読み込み失敗")
                return False
            
            # 現在の画像に黒い領域があるかチェック
            has_black, black_ratio = self.has_black_regions(image)
            
            # 厳密な顔検出
            is_face, face_region = self.is_face_image_strict(image)
            
            # バックアップ作成
            backup_path = self.backup_dir / img_path.name
            if not backup_path.exists():
                shutil.copy2(img_path, backup_path)
            
            if is_face and face_region:
                # 顔がある場合：適切なマスクを適用
                if has_black and black_ratio > 0.1:
                    # 既にマスクがある場合は、元の画像から再マスク
                    if backup_path.exists():
                        original_image = cv2.imread(str(backup_path))
                        if original_image is not None:
                            masked_image = self.apply_optimal_mask(original_image, face_region)
                        else:
                            masked_image = self.apply_optimal_mask(image, face_region)
                    else:
                        masked_image = self.apply_optimal_mask(image, face_region)
                else:
                    # マスクがない場合は新規適用
                    masked_image = self.apply_optimal_mask(image, face_region)
                
                cv2.imwrite(str(img_path), masked_image)
                self.face_images.append(str(img_path))
                print(f"👤 顔面画像: T字マスク適用")
                
            else:
                # 顔がない場合：マスクを除去
                if has_black and black_ratio > 0.05:
                    # 不正なマスクがある場合は元の画像に戻す
                    if backup_path.exists():
                        original_image = cv2.imread(str(backup_path))
                        if original_image is not None:
                            cv2.imwrite(str(img_path), original_image)
                            print(f"🫁 体幹部画像: 不正マスクを除去")
                        else:
                            print(f"🫁 体幹部画像: バックアップなし")
                    else:
                        print(f"🫁 体幹部画像: バックアップなし")
                else:
                    print(f"🫁 体幹部画像: マスクなし（正常）")
                
                self.trunk_images.append(str(img_path))
            
            self.fixed_count += 1
            return True
            
        except Exception as e:
            print(f"❌ 処理エラー: {e}")
            return False
    
    def fix_all_masks(self):
        """全BCC画像のマスクを修正"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # バックアップ作成
        if not self.create_backup():
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        print(f"🔧 BCC画像マスク修正開始")
        print(f"📁 対象画像数: {len(bcc_files)}枚")
        print(f"💾 バックアップ先: {self.backup_dir}")
        print("=" * 50)
        
        for i, img_path in enumerate(bcc_files, 1):
            print(f"[{i}/{len(bcc_files)}] ", end="")
            self.fix_image_mask(img_path)
            
            # 100枚ごとに進行状況を表示
            if i % 100 == 0:
                print(f"\n--- 進行状況 ({i}枚処理) ---")
                print(f"顔面画像: {len(self.face_images)}枚")
                print(f"体幹部画像: {len(self.trunk_images)}枚")
                print("-" * 30)
        
        # 結果の保存
        result = {
            'total_files': len(bcc_files),
            'fixed_files': self.fixed_count,
            'face_images': len(self.face_images),
            'trunk_images': len(self.trunk_images),
            'face_image_list': self.face_images,
            'trunk_image_list': self.trunk_images
        }
        
        with open('mask_fix_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 50)
        print(f"📊 マスク修正結果:")
        print(f"  総ファイル数: {len(bcc_files)}枚")
        print(f"  処理ファイル数: {self.fixed_count}枚")
        print(f"  顔面画像（マスク適用）: {len(self.face_images)}枚")
        print(f"  体幹部画像（マスクなし）: {len(self.trunk_images)}枚")
        print(f"💾 結果保存: mask_fix_result.json")
        print(f"💾 バックアップ: {self.backup_dir}/")
        
        return True

def main():
    print("🔧 BCC画像マスク修正ツール")
    print("=" * 50)
    print("👤 顔面画像: 最適T字マスクを適用")
    print("🫁 体幹部画像: 不正マスクを除去")
    print("💾 全画像のバックアップを作成")
    
    fixer = MaskFixer()
    
    print(f"\n処理を開始しますか？ (y/n): ", end="")
    try:
        response = input().lower()
        if response == 'y':
            if fixer.fix_all_masks():
                print(f"\n🎉 マスク修正完了!")
                print(f"\n💡 次のステップ:")
                print(f"1. マスク結果の確認")
                print(f"2. アノテーション作業の続行")
            else:
                print(f"\n❌ マスク修正でエラーが発生しました")
        else:
            print("キャンセルされました")
    except KeyboardInterrupt:
        print("\n中止されました")

if __name__ == "__main__":
    main()