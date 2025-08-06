#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顔の中心部をT字型にマスクするスクリプト
額、頬、顎は残し、目・鼻・口の部分をT字型に黒く塗りつぶす
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class TShapeFaceMasker:
    """顔の中心部をT字型にマスクするクラス"""
    
    def __init__(self):
        # OpenCVのHaar Cascade分類器を使用
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_face_features(self, image):
        """顔と目を検出してT字型マスクの領域を計算"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # 最大の顔を選択
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # 顔領域内で目を検出
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.05, 3)
        
        # T字型マスクの定義
        # 横棒：目から眉毛を含む領域
        # 縦棒：鼻から口を含む領域
        
        # デフォルト値（顔の比率に基づく）
        # 横棒（眉毛から目を含む大きめの領域）
        horizontal_top = y + int(h * 0.15)     # 眉毛の十分上から
        horizontal_bottom = y + int(h * 0.5)   # 目の十分下まで
        horizontal_left = x + int(w * 0.05)    # ほぼ顔の端から端まで
        horizontal_right = x + int(w * 0.95)   # ほぼ顔の端から端まで
        
        # 縦棒（鼻から口の領域）
        vertical_top = y + int(h * 0.4)        # 目の少し上から開始（眉毛と目の間）
        vertical_bottom = y + int(h * 0.9)     # 顎の少し上まで
        vertical_left = x + int(w * 0.25)      # 顔の中央部分（より広く）
        vertical_right = x + int(w * 0.75)     # 顔の中央部分（より広く）
        
        # 目が検出された場合は、より正確な位置を使用
        if len(eyes) >= 2:
            # 目の位置を顔全体の座標系に変換
            eyes_global = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            # 目の位置から横棒の範囲を調整
            eye_tops = [ey for ex, ey, ew, eh in eyes_global]
            eye_bottoms = [ey + eh for ex, ey, ew, eh in eyes_global]
            
            # 横棒：眉毛の十分上から目の十分下まで（大きめに）
            horizontal_top = min(eye_tops) - 40  # 眉毛の上部を確実に含む
            horizontal_bottom = max(eye_bottoms) + 20  # 目の下を十分含む
            
            # 横棒の左右：顔の幅ほぼ全体
            eye_lefts = [ex for ex, ey, ew, eh in eyes_global]
            eye_rights = [ex + ew for ex, ey, ew, eh in eyes_global]
            
            # 顔の端近くまで広げる
            horizontal_left = x + int(w * 0.05)  # 顔の左端から5%内側
            horizontal_right = x + int(w * 0.95)  # 顔の右端から5%内側
            
            # 縦棒は眉毛と目の間から開始
            vertical_top = min(eye_tops) - 10  # 眉毛の下あたりから
        
        return {
            'horizontal': {
                'top': horizontal_top,
                'bottom': horizontal_bottom,
                'left': horizontal_left,
                'right': horizontal_right
            },
            'vertical': {
                'top': vertical_top,
                'bottom': vertical_bottom,
                'left': vertical_left,
                'right': vertical_right
            }
        }
    
    def create_t_shape_mask(self, image_shape, regions):
        """T字型のマスクを作成"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if regions is None:
            return mask
        
        # 横棒を描画
        horizontal = regions['horizontal']
        cv2.rectangle(mask, 
                     (horizontal['left'], horizontal['top']),
                     (horizontal['right'], horizontal['bottom']),
                     255, -1)
        
        # 縦棒を描画
        vertical = regions['vertical']
        cv2.rectangle(mask, 
                     (vertical['left'], vertical['top']),
                     (vertical['right'], vertical['bottom']),
                     255, -1)
        
        return mask
    
    def mask_face_t_shape(self, image_path, output_path=None, preview=False):
        """画像の顔中心部をT字型にマスク"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # 顔の特徴を検出
        regions = self.detect_face_features(image)
        if regions is None:
            print(f"顔を検出できませんでした: {image_path}")
            return False
        
        if preview:
            # プレビューモード：マスク範囲を赤い枠で表示
            preview_image = image.copy()
            
            # 横棒を赤い枠で描画
            horizontal = regions['horizontal']
            cv2.rectangle(preview_image, 
                         (horizontal['left'], horizontal['top']),
                         (horizontal['right'], horizontal['bottom']),
                         (0, 0, 255), 2)
            
            # 縦棒を赤い枠で描画
            vertical = regions['vertical']
            cv2.rectangle(preview_image, 
                         (vertical['left'], vertical['top']),
                         (vertical['right'], vertical['bottom']),
                         (0, 0, 255), 2)
            
            # T字の説明テキストを追加
            cv2.putText(preview_image, "T-shape mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if output_path is None:
                output_path = Path(image_path).parent / f"preview_t_{Path(image_path).name}"
            cv2.imwrite(str(output_path), preview_image)
        else:
            # 通常モード：マスクを適用（黒く塗りつぶす）
            mask = self.create_t_shape_mask(image.shape, regions)
            masked_image = image.copy()
            masked_image[mask > 0] = 0  # マスク領域を黒に
            
            if output_path is None:
                output_path = Path(image_path).parent / f"masked_t_{Path(image_path).name}"
            cv2.imwrite(str(output_path), masked_image)
        
        return True
    
    def process_directory(self, input_dir, output_dir, preview=False):
        """ディレクトリ内のすべての画像を処理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 画像ファイルを収集（サブディレクトリも含む）
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            image_files.extend(input_path.rglob(ext))
        
        print(f"処理対象: {len(image_files)}枚の画像")
        if preview:
            print("プレビューモード: T字型マスク範囲を赤い枠で表示します")
        else:
            print("T字型マスクモード: 額・頬・顎を残し、中心部をT字型にマスクします")
        
        success_count = 0
        failed_files = []
        
        for image_path in tqdm(image_files, desc="処理中"):
            # サブディレクトリ構造を保持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            if self.mask_face_t_shape(image_path, output_file, preview):
                success_count += 1
            else:
                failed_files.append(str(image_path))
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        if failed_files:
            # 失敗したファイルをJSONで保存
            with open(output_path / 'failed_files.json', 'w') as f:
                json.dump(failed_files, f, indent=2, ensure_ascii=False)
            
            print(f"\n失敗したファイル（最初の10個）:")
            for f in failed_files[:10]:
                print(f"  - {f}")
            
            if len(failed_files) > 10:
                print(f"  ... 他 {len(failed_files) - 10} ファイル")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='顔の中心部をT字型に黒く塗りつぶす（額・頬・顎は残す）')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：T字型マスク範囲を赤い枠で表示')
    
    args = parser.parse_args()
    
    masker = TShapeFaceMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_t_shape(input_path, output_path, args.preview):
            mode = "プレビュー" if args.preview else "マスク"
            print(f"{mode}処理完了: {output_path or f'{"preview_t" if args.preview else "masked_t"}_{input_path.name}'}")
        else:
            print("処理に失敗しました")
    
    elif input_path.is_dir():
        # ディレクトリの処理
        suffix = '_t_preview' if args.preview else '_t_masked'
        output_dir = args.output if args.output else str(input_path) + suffix
        masker.process_directory(input_path, output_dir, args.preview)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == '__main__':
    main()