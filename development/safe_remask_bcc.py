#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
安全なBCC再マスク処理（エラーハンドリング強化版）
破損ファイルのスキップと進行状況の保存機能付き
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time

class SafeBCCRemasker:
    def __init__(self):
        # OpenCV顔検出器とパーツ検出器を初期化
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # BCC画像パスを設定
        self.bcc_dir = Path('organized_advanced_masked/Basal cell carcinoma')
        
        self.processed_files = []
        self.face_files = []
        self.trunk_files = []
        self.error_files = []
        
        # 進行状況ファイル
        self.progress_file = 'bcc_remask_progress.json'
        
    def load_progress(self):
        """進行状況を読み込み"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                processed_set = set(progress.get('processed_files', []))
                return processed_set
            except Exception as e:
                print(f"⚠️ 進行状況読み込みエラー: {e}")
        return set()
    
    def save_progress(self):
        """進行状況を保存"""
        progress = {
            'processed_files': self.processed_files,
            'face_files': self.face_files,
            'trunk_files': self.trunk_files,
            'error_files': self.error_files,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 進行状況保存エラー: {e}")
    
    def detect_face_features_opencv(self, image):
        """OpenCVでの顔特徴検出（安全版）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 顔検出
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, False
            
            # 最大の顔を選択
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # 顔が小さすぎる場合はスキップ
            if w < 50 or h < 50:
                return None, False
            
            face_roi = gray[y:y+h, x:x+w]
            
            # 目検出
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(10, 10))
            
            # T字マスクの範囲を計算
            face_center_x = x + w // 2
            face_top = y
            face_bottom = y + h
            
            # 眉毛領域（organized_advanced_maskedスタイル）
            eyebrow_top = max(0, face_top - int(h * 0.2))
            eyebrow_bottom = y + int(h * 0.35)
            
            # 鼻と口の範囲
            nose_top = eyebrow_bottom
            nose_bottom = y + int(h * 0.75)
            mouth_top = y + int(h * 0.6)
            mouth_bottom = face_bottom - int(h * 0.05)
            
            # T字マスクの領域を定義
            mask_regions = {
                'horizontal': {
                    'top': eyebrow_top,
                    'bottom': eyebrow_bottom,
                    'left': x + int(w * 0.1),
                    'right': x + w - int(w * 0.1)
                },
                'vertical_nose': {
                    'top': nose_top,
                    'bottom': nose_bottom,
                    'left': face_center_x - int(w * 0.15),
                    'right': face_center_x + int(w * 0.15)
                },
                'vertical_mouth': {
                    'top': mouth_top,
                    'bottom': mouth_bottom,
                    'left': face_center_x - int(w * 0.2),
                    'right': face_center_x + int(w * 0.2)
                }
            }
            
            return mask_regions, True
            
        except Exception as e:
            print(f"❌ 顔検出エラー: {e}")
            return None, False
    
    def create_optimal_t_mask(self, image_shape, mask_regions):
        """最適なT字マスクを作成"""
        if mask_regions is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 安全な座標クリッピング
        def safe_coords(coords):
            return max(0, min(coords, w-1 if coords == coords else h-1))
        
        try:
            # 横棒（眉毛から目の領域）
            horizontal = mask_regions['horizontal']
            cv2.rectangle(mask, 
                         (safe_coords(horizontal['left']), safe_coords(horizontal['top'])),
                         (safe_coords(horizontal['right']), safe_coords(horizontal['bottom'])),
                         255, -1)
            
            # 縦棒（鼻の領域）
            nose = mask_regions['vertical_nose']
            cv2.rectangle(mask, 
                         (safe_coords(nose['left']), safe_coords(nose['top'])),
                         (safe_coords(nose['right']), safe_coords(nose['bottom'])),
                         255, -1)
            
            # 口の領域
            mouth = mask_regions['vertical_mouth']
            cv2.rectangle(mask, 
                         (safe_coords(mouth['left']), safe_coords(mouth['top'])),
                         (safe_coords(mouth['right']), safe_coords(mouth['bottom'])),
                         255, -1)
        
        except Exception as e:
            print(f"❌ マスク作成エラー: {e}")
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        return mask
    
    def process_bcc_image_safe(self, image_path):
        """BCC画像を安全に処理"""
        try:
            print(f"📸 処理中: {image_path.name}")
            
            # 画像読み込み（エラーハンドリング強化）
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"❌ 画像読み込み失敗: {image_path}")
                self.error_files.append(str(image_path))
                return False
            
            # 画像サイズの確認
            if image.shape[0] < 50 or image.shape[1] < 50:
                print(f"❌ 画像が小さすぎます: {image_path}")
                self.error_files.append(str(image_path))
                return False
            
            # 顔特徴検出
            mask_regions, is_face = self.detect_face_features_opencv(image)
            
            if is_face and mask_regions:
                print(f"👤 顔面画像: マスク処理実行")
                
                # 最適なT字マスクを作成
                mask = self.create_optimal_t_mask(image.shape, mask_regions)
                
                # マスクを適用
                masked_image = image.copy()
                masked_image[mask > 0] = [0, 0, 0]
                
                # 安全な保存
                success = cv2.imwrite(str(image_path), masked_image)
                if not success:
                    print(f"❌ 画像保存失敗: {image_path}")
                    self.error_files.append(str(image_path))
                    return False
                
                self.face_files.append(str(image_path))
                print(f"✅ T字マスク適用完了: {image_path.name}")
                return True
            else:
                print(f"🫁 体幹部画像: マスク不要（スキップ）")
                self.trunk_files.append(str(image_path))
                return True
                
        except Exception as e:
            print(f"❌ 処理エラー {image_path.name}: {e}")
            self.error_files.append(str(image_path))
            return False
    
    def remask_bcc_batch(self, batch_size=50):
        """バッチ処理でBCC画像を再マスク"""
        if not self.bcc_dir.exists():
            print(f"❌ BCCフォルダが見つかりません: {self.bcc_dir}")
            return False
        
        # BCC画像ファイルを取得
        bcc_files = list(self.bcc_dir.glob("*.jpg")) + list(self.bcc_dir.glob("*.JPG"))
        
        # 進行状況を読み込み
        processed_set = self.load_progress()
        
        # 未処理ファイルをフィルタリング
        remaining_files = [f for f in bcc_files if str(f) not in processed_set]
        
        print(f"🔍 BCC画像の安全な再マスク処理開始")
        print(f"📁 総ファイル数: {len(bcc_files)}枚")
        print(f"📁 残りファイル数: {len(remaining_files)}枚")
        print(f"🔄 バッチサイズ: {batch_size}枚")
        print("=" * 60)
        
        success_count = 0
        
        # バッチ処理
        for i in range(0, len(remaining_files), batch_size):
            batch = remaining_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_files) + batch_size - 1) // batch_size
            
            print(f"\n🔄 バッチ {batch_num}/{total_batches} 処理中 ({len(batch)}枚)")
            
            for j, img_path in enumerate(batch, 1):
                print(f"[{i+j}/{len(remaining_files)}] ", end="")
                
                if self.process_bcc_image_safe(img_path):
                    success_count += 1
                    self.processed_files.append(str(img_path))
                
                # 5枚ごとに進行状況を保存
                if (i + j) % 5 == 0:
                    self.save_progress()
            
            # バッチ完了後に進行状況を保存
            self.save_progress()
            print(f"✅ バッチ {batch_num} 完了")
            
            # 少し待機（システム負荷軽減）
            time.sleep(0.5)
        
        # 最終結果の保存
        result = {
            'total_files': len(bcc_files),
            'processed_files': len(self.processed_files),
            'face_files': len(self.face_files),
            'trunk_files': len(self.trunk_files),
            'error_files': len(self.error_files),
            'success_rate': success_count / len(remaining_files) if remaining_files else 1.0
        }
        
        with open('bcc_safe_remask_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"📊 BCC安全再マスク処理結果:")
        print(f"  総ファイル数: {len(bcc_files)}枚")
        print(f"  処理成功: {success_count}枚")
        print(f"  顔面画像（マスク適用）: {len(self.face_files)}枚")
        print(f"  体幹部画像（スキップ）: {len(self.trunk_files)}枚")
        print(f"  エラーファイル: {len(self.error_files)}枚")
        print(f"💾 結果保存: bcc_safe_remask_result.json")
        
        if self.error_files:
            print(f"\n⚠️ エラーファイル（最初の5個）:")
            for error_file in self.error_files[:5]:
                print(f"  - {Path(error_file).name}")
        
        return success_count > 0

def main():
    print("🎭 organized_advanced_masked BCC安全再マスク処理")
    print("=" * 60)
    print("📌 エラーハンドリング強化・バッチ処理・進行状況保存機能付き")
    print("🎯 顔面画像のみ処理、体幹部画像はスキップ")
    
    remasker = SafeBCCRemasker()
    
    # バッチサイズを設定（デフォルト50枚）
    batch_size = 30  # 少し小さめに設定
    
    print(f"\n開始しますか？ (Enter=開始, Ctrl+C=中止)")
    try:
        input()
    except KeyboardInterrupt:
        print("\n中止されました")
        return
    
    # 安全な再マスク処理実行
    if remasker.remask_bcc_batch(batch_size):
        print(f"\n🎉 BCC安全再マスク処理完了!")
        
        # 進行状況ファイルを削除（完了のため）
        try:
            Path(remasker.progress_file).unlink()
            print(f"📁 進行状況ファイルを削除")
        except:
            pass
        
        print(f"\n💡 次のステップ:")
        print(f"1. アノテーション作業の続行")
        print(f"2. マスク品質の確認")
    else:
        print(f"\n❌ 再マスク処理でエラーが発生しました")
        print(f"📄 進行状況は {remasker.progress_file} に保存されています")

if __name__ == "__main__":
    main()