#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
organized_advanced_maskedフォルダのBCC画像を最適なマスク処理で再マスク
眉毛から目、鼻、口にかけてのT字マスクをOpenCVで実装
顔面画像のみに適用し、体幹部画像はスキップ
"""

import cv2
import numpy as np
from pathlib import Path
import json

class OrganizedBCCRemasker:
    def __init__(self):
        # OpenCV顔検出器とパーツ検出器を初期化
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        
        self.processed_files = []
        self.face_files = []
        self.trunk_files = []
        
    def detect_face_features_opencv(self, image):
        """OpenCVでの顔特徴検出（眉毛から口まで最適マスク）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, False
        
        # 最大の顔を選択
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        face_roi = gray[y:y+h, x:x+w]
        
        # 目検出
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(10, 10))
        
        if len(eyes) < 2:
            return None, False
        
        # 目を絶対座標に変換
        abs_eyes = []
        for (ex, ey, ew, eh) in eyes:
            abs_eyes.append((x + ex, y + ey, ew, eh))
        
        # 眉毛から口までの範囲を計算（organized_advanced_maskedと同等の最適マスク）
        face_center_x = x + w // 2
        face_top = y
        face_bottom = y + h
        
        # 眉毛領域（顔の上部15%から目の上まで）
        eyebrow_top = max(0, face_top - int(h * 0.15))  # 眉毛の上部（余裕を持って）
        
        # 目の位置から眉毛下部を推定
        if abs_eyes:
            eye_y_positions = [ey for (_, ey, _, _) in abs_eyes]
            highest_eye = min(eye_y_positions)
            eyebrow_bottom = highest_eye - 5  # 目の少し上
        else:
            eyebrow_bottom = y + int(h * 0.3)
        
        # 鼻の範囲（目の下から顔の中央部）
        nose_top = eyebrow_bottom + 5
        nose_bottom = y + int(h * 0.75)
        nose_left = face_center_x - int(w * 0.15)
        nose_right = face_center_x + int(w * 0.15)
        
        # 口の範囲（顔の下部25%）
        mouth_top = y + int(h * 0.65)
        mouth_bottom = face_bottom - int(h * 0.05)
        mouth_left = face_center_x - int(w * 0.2)
        mouth_right = face_center_x + int(w * 0.2)
        
        # T字マスクの領域を定義
        mask_regions = {
            # 横棒：眉毛から目の領域（顔幅の90%）
            'horizontal': {
                'top': eyebrow_top,
                'bottom': eyebrow_bottom + int(h * 0.1),  # 目の下まで少し余裕
                'left': x + int(w * 0.05),  # 顔幅の5%内側から
                'right': x + w - int(w * 0.05)  # 顔幅の5%内側まで
            },
            # 縦棒：鼻から口の領域
            'vertical_nose': {
                'top': nose_top,
                'bottom': nose_bottom,
                'left': nose_left,
                'right': nose_right
            },
            # 口の領域
            'vertical_mouth': {
                'top': mouth_top,
                'bottom': mouth_bottom,
                'left': mouth_left,
                'right': mouth_right
            }
        }
        
        return mask_regions, True
    
    def create_optimal_t_mask(self, image_shape, mask_regions):
        """最適なT字マスクを作成（眉毛から口まで）"""
        if mask_regions is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 横棒（眉毛から目の領域）を描画
        horizontal = mask_regions['horizontal']
        cv2.rectangle(mask, 
                     (horizontal['left'], horizontal['top']),
                     (horizontal['right'], horizontal['bottom']),
                     255, -1)
        
        # 縦棒（鼻の領域）を描画
        nose = mask_regions['vertical_nose']
        cv2.rectangle(mask, 
                     (nose['left'], nose['top']),
                     (nose['right'], nose['bottom']),
                     255, -1)
        
        # 口の領域を描画
        mouth = mask_regions['vertical_mouth']
        cv2.rectangle(mask, 
                     (mouth['left'], mouth['top']),
                     (mouth['right'], mouth['bottom']),
                     255, -1)
        
        return mask
    
    def process_bcc_image(self, image_path):
        """BCC画像を処理（顔面のみマスク、体幹部はスキップ）"""
        print(f"📸 処理中: {image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 画像読み込み失敗: {image_path}")
            return False
        
        # 顔特徴検出
        mask_regions, is_face = self.detect_face_features_opencv(image)
        
        if is_face and mask_regions:
            print(f"👤 顔面画像: マスク処理実行")
            
            # 最適なT字マスクを作成
            mask = self.create_optimal_t_mask(image.shape, mask_regions)
            
            # マスクを適用（黒く塗りつぶし）
            masked_image = image.copy()
            masked_image[mask > 0] = [0, 0, 0]
            
            # 元の画像を上書き
            cv2.imwrite(str(image_path), masked_image)
            
            self.face_files.append(str(image_path))
            print(f"✅ T字マスク適用完了: {image_path.name}")
            return True
        else:
            print(f"🫁 体幹部画像: マスク不要（スキップ）")
            self.trunk_files.append(str(image_path))
            return True
    
    def remask_all_bcc_images(self):
        """organized_advanced_maskedフォルダのBCC画像を全て再マスク"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        print(f"🔍 BCC画像の最適マスク再適用開始")
        print(f"📁 対象画像数: {len(bcc_files)}枚")
        print(f"🎭 organized_advanced_maskedと同等の最適マスク処理")
        print("=" * 60)
        
        success_count = 0
        
        for i, img_path in enumerate(bcc_files, 1):
            print(f"\n[{i}/{len(bcc_files)}] ", end="")
            
            if self.process_bcc_image(img_path):
                success_count += 1
                self.processed_files.append(str(img_path))
        
        # 結果の保存
        result = {
            'total_files': len(bcc_files),
            'processed_files': len(self.processed_files),
            'face_files': len(self.face_files),
            'trunk_files': len(self.trunk_files),
            'success_rate': success_count / len(bcc_files) if bcc_files else 0,
            'face_files_list': self.face_files,
            'trunk_files_list': self.trunk_files
        }
        
        with open('bcc_remask_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"📊 BCC再マスク処理結果:")
        print(f"  総ファイル数: {len(bcc_files)}枚")
        print(f"  処理成功: {success_count}枚")
        print(f"  顔面画像（マスク適用）: {len(self.face_files)}枚")
        print(f"  体幹部画像（スキップ）: {len(self.trunk_files)}枚")
        print(f"💾 結果保存: bcc_remask_result.json")
        
        return success_count > 0
    
    def preview_mask_result(self, num_preview=3):
        """マスク結果のプレビュー表示"""
        if not self.face_files:
            print("👤 マスク処理された顔面画像がありません")
            return
        
        print(f"\n🔍 マスク結果プレビュー（最初の{min(num_preview, len(self.face_files))}枚）")
        
        for i, file_path in enumerate(self.face_files[:num_preview]):
            img_path = Path(file_path)
            image = cv2.imread(str(img_path))
            
            if image is not None:
                # リサイズして表示
                height, width = image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = 800
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                window_name = f"Remasked BCC {i+1} - {img_path.name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, image)
                
                print(f"📸 プレビュー表示中: {img_path.name}")
                print("何かキーを押すと次へ...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

def main():
    print("🎭 organized_advanced_masked BCC最適再マスク処理")
    print("=" * 60)
    print("📌 眉毛から目、鼻、口までの最適T字マスクを再適用")
    print("🎯 顔面画像のみ処理、体幹部画像はスキップ")
    
    remasker = OrganizedBCCRemasker()
    
    # 再マスク処理実行
    if remasker.remask_all_bcc_images():
        print(f"\n🎉 BCC最適再マスク処理完了!")
        
        # プレビュー表示
        print(f"\nプレビューを表示しますか？ (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                remasker.preview_mask_result()
        except:
            pass
        
        print(f"\n💡 次のステップ:")
        print(f"1. アノテーション作業の続行")
        print(f"2. マスク品質の確認")
    else:
        print(f"\n❌ 再マスク処理でエラーが発生しました")

if __name__ == "__main__":
    main()