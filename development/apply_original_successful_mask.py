#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
以前成功していたマスク処理をorganizedフォルダ全体に適用
BCC追加前と同じ品質のマスクを再現
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import json
import time

class OriginalSuccessfulMasker:
    def __init__(self):
        # OpenCV顔検出器を初期化（以前成功していた設定）
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.source_dir = Path('organized')
        self.target_dir = Path('organized_original_masked')
        
        self.processed_count = 0
        self.face_masked_count = 0
        self.trunk_copied_count = 0
        self.error_count = 0
        
    def detect_face_conservative(self, image):
        """保守的な顔検出（以前成功していた方法）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 以前成功していた保守的なパラメータ
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,  # より厳格
                minSize=(60, 60),  # より大きな最小サイズ
                maxSize=(300, 300)  # 最大サイズ制限
            )
            
            if len(faces) > 0:
                # 最大の顔を選択
                best_face = max(faces, key=lambda x: x[2] * x[3])
                return tuple(best_face)
            
            return None
            
        except Exception as e:
            print(f"❌ 顔検出エラー: {e}")
            return None
    
    def apply_original_t_mask(self, image, face_region):
        """以前成功していたT字マスク（保守的なサイズ）"""
        try:
            x, y, w, h = face_region
            face_center_x = x + w // 2
            
            # 以前成功していた保守的なマスクサイズ
            # 眉毛から目の領域（横棒）- 保守的
            eyebrow_top = max(0, y - int(h * 0.2))  # 小さめの余裕
            eye_bottom = y + int(h * 0.4)  # 控えめ
            horizontal_left = max(0, x)  # 顔の端から
            horizontal_right = min(image.shape[1], x + w)  # 顔の端まで
            
            # 鼻の領域 - 保守的
            nose_top = y + int(h * 0.25)
            nose_bottom = y + int(h * 0.7)
            nose_left = max(0, face_center_x - int(w * 0.15))
            nose_right = min(image.shape[1], face_center_x + int(w * 0.15))
            
            # 口の領域 - 保守的
            mouth_top = y + int(h * 0.55)
            mouth_bottom = min(image.shape[0], y + h)
            mouth_left = max(0, face_center_x - int(w * 0.2))
            mouth_right = min(image.shape[1], face_center_x + int(w * 0.2))
            
            # マスク適用
            masked = image.copy()
            
            # 横棒（眉毛から目）
            cv2.rectangle(masked, (horizontal_left, eyebrow_top), 
                         (horizontal_right, eye_bottom), (0, 0, 0), -1)
            
            # 鼻の領域
            cv2.rectangle(masked, (nose_left, nose_top), 
                         (nose_right, nose_bottom), (0, 0, 0), -1)
            
            # 口の領域
            cv2.rectangle(masked, (mouth_left, mouth_top), 
                         (mouth_right, mouth_bottom), (0, 0, 0), -1)
            
            return masked
            
        except Exception as e:
            print(f"❌ マスク適用エラー: {e}")
            return image
    
    def process_image(self, source_path, target_path):
        """画像を処理（以前成功していた方法）"""
        try:
            # 画像読み込み
            image = cv2.imread(str(source_path))
            if image is None:
                print(f"❌ 読み込み失敗: {source_path.name}")
                self.error_count += 1
                return False
            
            # 保守的な顔検出
            face_region = self.detect_face_conservative(image)
            
            if face_region:
                # 顔面画像：保守的なT字マスクを適用
                masked_image = self.apply_original_t_mask(image, face_region)
                cv2.imwrite(str(target_path), masked_image)
                print(f"👤 顔面マスク: {source_path.name}")
                self.face_masked_count += 1
            else:
                # 体幹部画像：そのままコピー
                shutil.copy2(source_path, target_path)
                print(f"🫁 体幹部コピー: {source_path.name}")
                self.trunk_copied_count += 1
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            print(f"❌ 処理エラー {source_path.name}: {e}")
            self.error_count += 1
            return False
    
    def process_all_classes(self):
        """全クラスの画像を処理（以前成功していた方法）"""
        if not self.source_dir.exists():
            print(f"❌ ソースフォルダが見つかりません: {self.source_dir}")
            return False
        
        # ターゲットディレクトリを作成
        self.target_dir.mkdir(exist_ok=True)
        
        print(f"🎭 以前成功していたマスク処理を再現")
        print(f"📂 ソース: {self.source_dir}")
        print(f"📂 ターゲット: {self.target_dir}")
        print(f"🎯 保守的なT字マスク: 眉毛・目・鼻・口を適度にカバー")
        print("=" * 60)
        
        classes = ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 'Nevus', 
                  'Basal cell carcinoma', 'Seborrheic keratosis', 'Malignant melanoma']
        
        total_files = 0
        
        # 各クラスを処理
        for class_name in classes:
            source_class_dir = self.source_dir / class_name
            target_class_dir = self.target_dir / class_name
            
            if not source_class_dir.exists():
                print(f"⚠️ クラスフォルダなし: {class_name}")
                continue
            
            # ターゲットクラスディレクトリを作成
            target_class_dir.mkdir(exist_ok=True)
            
            # 画像ファイルを取得
            image_files = list(source_class_dir.glob("*.jpg")) + list(source_class_dir.glob("*.JPG"))
            
            if not image_files:
                print(f"📁 {class_name}: 画像なし")
                continue
            
            print(f"\n📁 {class_name}: {len(image_files)}枚 処理中...")
            total_files += len(image_files)
            
            # 各画像を処理
            for i, source_path in enumerate(image_files, 1):
                target_path = target_class_dir / source_path.name
                
                print(f"[{i}/{len(image_files)}] ", end="")
                self.process_image(source_path, target_path)
                
                # 50枚ごとに進行状況表示
                if i % 50 == 0:
                    print(f"\n--- {class_name} 進行状況 ({i}/{len(image_files)}) ---")
                    print(f"顔面マスク: {self.face_masked_count}枚")
                    print(f"体幹部コピー: {self.trunk_copied_count}枚")
                    print(f"エラー: {self.error_count}枚")
        
        # 結果保存
        result = {
            'total_files': total_files,
            'processed_files': self.processed_count,
            'face_masked': self.face_masked_count,
            'trunk_copied': self.trunk_copied_count,
            'error_files': self.error_count,
            'success_rate': self.processed_count / total_files if total_files > 0 else 0
        }
        
        with open('original_mask_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"📊 以前成功していたマスク処理結果:")
        print(f"  総ファイル数: {total_files}枚")
        print(f"  処理成功: {self.processed_count}枚")
        print(f"  顔面マスク適用: {self.face_masked_count}枚")
        print(f"  体幹部コピー: {self.trunk_copied_count}枚")
        print(f"  エラー: {self.error_count}枚")
        print(f"  成功率: {result['success_rate']*100:.1f}%")
        print(f"💾 結果保存: original_mask_result.json")
        print(f"📂 出力先: {self.target_dir}/")
        
        return self.processed_count > 0

def main():
    print("🎭 以前成功していたマスク処理を再現")
    print("=" * 60)
    print("📌 organizedフォルダの全画像を処理")
    print("👤 顔面画像: 保守的なT字マスク適用")
    print("🫁 体幹部画像: そのままコピー")
    print("📂 出力: organized_original_maskedフォルダ")
    
    masker = OriginalSuccessfulMasker()
    
    print(f"\n🔄 処理を開始します...")
    
    if masker.process_all_classes():
        print(f"\n🎉 以前成功していたマスク処理完了!")
        print(f"\n💡 次のステップ:")
        print(f"1. organized_original_maskedフォルダの確認")
        print(f"2. マスク品質の比較")
    else:
        print(f"\n❌ 処理でエラーが発生しました")

if __name__ == "__main__":
    main()