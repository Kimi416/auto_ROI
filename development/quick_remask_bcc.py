#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高速BCC再マスク処理（軽量版）
小さなバッチサイズで安全かつ高速に処理
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time

class QuickBCCRemasker:
    def __init__(self):
        # OpenCV顔検出器とパーツ検出器を初期化
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        
        self.face_count = 0
        self.trunk_count = 0
        self.error_count = 0
        
    def detect_and_mask_face(self, image):
        """顔検出とマスク処理を一度に実行"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 顔検出
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
            
            if len(faces) == 0:
                return image, False  # 体幹部画像（マスク不要）
            
            # 最大の顔を選択
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # T字マスクの領域を計算（organized_advanced_maskedスタイル）
            face_center_x = x + w // 2
            
            # 眉毛から目の領域（横棒）
            eyebrow_top = max(0, y - int(h * 0.25))  # 眉毛の上部（大きめの余裕）
            eye_bottom = y + int(h * 0.45)  # 目の下部
            horizontal_left = x + int(w * 0.05)  # 顔幅の5%内側
            horizontal_right = x + w - int(w * 0.05)  # 顔幅の5%内側
            
            # 鼻の領域（縦棒上部）
            nose_top = y + int(h * 0.3)
            nose_bottom = y + int(h * 0.7)
            nose_left = face_center_x - int(w * 0.12)
            nose_right = face_center_x + int(w * 0.12)
            
            # 口の領域（縦棒下部）
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
            
            return masked, True  # 顔面画像（マスク適用済み）
            
        except Exception as e:
            print(f"❌ マスク処理エラー: {e}")
            return image, False
    
    def process_bcc_batch_quick(self, batch_size=10):
        """小さなバッチで高速処理"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        print(f"🔍 BCC画像の高速再マスク処理開始")
        print(f"📁 総ファイル数: {len(bcc_files)}枚")
        print(f"🔄 バッチサイズ: {batch_size}枚")
        print("=" * 50)
        
        start_time = time.time()
        
        # 小バッチで処理
        for i in range(0, len(bcc_files), batch_size):
            batch = bcc_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(bcc_files) + batch_size - 1) // batch_size
            
            print(f"\n🔄 バッチ {batch_num}/{total_batches} ({len(batch)}枚)")
            
            for j, img_path in enumerate(batch):
                file_num = i + j + 1
                print(f"[{file_num}/{len(bcc_files)}] {img_path.name}", end=" ... ")
                
                try:
                    # 画像読み込み
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print("❌ 読み込み失敗")
                        self.error_count += 1
                        continue
                    
                    # 顔検出とマスク処理
                    masked_image, is_face = self.detect_and_mask_face(image)
                    
                    if is_face:
                        # 顔面画像の場合、マスク済み画像を保存
                        cv2.imwrite(str(img_path), masked_image)
                        print("👤 顔面マスク適用")
                        self.face_count += 1
                    else:
                        # 体幹部画像の場合、そのまま
                        print("🫁 体幹部（スキップ）")
                        self.trunk_count += 1
                        
                except Exception as e:
                    print(f"❌ エラー: {e}")
                    self.error_count += 1
            
            # バッチ完了の進行状況表示
            elapsed = time.time() - start_time
            remaining_files = len(bcc_files) - (i + len(batch))
            if file_num > 0:
                avg_time_per_file = elapsed / file_num
                estimated_remaining = avg_time_per_file * remaining_files
                print(f"⏱️ 経過時間: {elapsed:.1f}秒, 推定残り時間: {estimated_remaining:.1f}秒")
        
        # 結果の保存
        total_time = time.time() - start_time
        result = {
            'total_files': len(bcc_files),
            'face_files': self.face_count,
            'trunk_files': self.trunk_count,
            'error_files': self.error_count,
            'processing_time_seconds': total_time,
            'files_per_second': len(bcc_files) / total_time if total_time > 0 else 0
        }
        
        with open('quick_remask_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 50)
        print(f"📊 BCC高速再マスク処理結果:")
        print(f"  総ファイル数: {len(bcc_files)}枚")
        print(f"  顔面画像（マスク適用）: {self.face_count}枚")
        print(f"  体幹部画像（スキップ）: {self.trunk_count}枚")
        print(f"  エラーファイル: {self.error_count}枚")
        print(f"  処理時間: {total_time:.1f}秒")
        print(f"  処理速度: {result['files_per_second']:.1f}枚/秒")
        print(f"💾 結果保存: quick_remask_result.json")
        
        return True

def main():
    print("🎭 BCC高速再マスク処理")
    print("=" * 50)
    print("⚡ 軽量・高速処理でorganized_advanced_maskedスタイルのマスクを適用")
    print("🎯 顔面画像のみ処理、体幹部画像はスキップ")
    
    remasker = QuickBCCRemasker()
    
    # 小さなバッチサイズで安全に処理
    batch_size = 10
    
    print(f"\n処理を開始します... (バッチサイズ: {batch_size})")
    
    if remasker.process_bcc_batch_quick(batch_size):
        print(f"\n🎉 BCC高速再マスク処理完了!")
        print(f"\n💡 次のステップ:")
        print(f"1. アノテーション作業の続行")
        print(f"2. マスク品質の確認")
    else:
        print(f"\n❌ 処理でエラーが発生しました")

if __name__ == "__main__":
    main()