#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
目と鼻の部分のみを黒く塗りつぶすスクリプト（より狭い範囲）
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class FocusedFaceMasker:
    """目と鼻の部分のみをマスクするクラス"""
    
    def __init__(self):
        # OpenCVのHaar Cascade分類器を使用
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_face_region(self, image):
        """顔と目を検出して目と鼻の領域を推定"""
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
        
        # デフォルト値（目が検出されない場合）
        # 目の領域：顔の上端から35%〜55%の位置
        mask_top = y + int(h * 0.35)
        mask_bottom = y + int(h * 0.55)
        
        # 左右の境界：目の幅程度（顔の幅の35%）
        face_center_x = x + w // 2
        mask_width = int(w * 0.35)
        left_boundary = face_center_x - mask_width // 2
        right_boundary = face_center_x + mask_width // 2
        
        # 目が検出された場合は、より正確な位置を使用
        if len(eyes) >= 2:
            # 目の位置を顔全体の座標系に変換
            eyes_global = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
            
            # 目の上端と下端を見つける
            eye_tops = [ey for ex, ey, ew, eh in eyes_global]
            eye_bottoms = [ey + eh for ex, ey, ew, eh in eyes_global]
            
            # マスク範囲：目の上端から目の下端の1.5倍下まで（鼻を含む）
            mask_top = min(eye_tops)
            eye_height = max(eye_bottoms) - min(eye_tops)
            mask_bottom = mask_top + int(eye_height * 2.5)
            
            # 左右の境界：両目の外端
            eye_lefts = [ex for ex, ey, ew, eh in eyes_global]
            eye_rights = [ex + ew for ex, ey, ew, eh in eyes_global]
            left_boundary = min(eye_lefts) - 10
            right_boundary = max(eye_rights) + 10
        
        return {
            'top': mask_top,
            'bottom': mask_bottom,
            'left': left_boundary,
            'right': right_boundary
        }
    
    def create_center_mask(self, image_shape, region):
        """マスクを作成"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if region is None:
            return mask
        
        # 矩形を描画
        cv2.rectangle(mask, 
                     (region['left'], region['top']),
                     (region['right'], region['bottom']),
                     255, -1)
        
        return mask
    
    def mask_face_center(self, image_path, output_path=None, preview=False):
        """画像の目と鼻の部分をマスク"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # 顔領域を検出
        region = self.detect_face_region(image)
        if region is None:
            print(f"顔を検出できませんでした: {image_path}")
            return False
        
        # マスクを作成
        mask = self.create_center_mask(image.shape, region)
        
        if preview:
            # プレビューモード：マスク範囲を赤い枠で表示
            preview_image = image.copy()
            cv2.rectangle(preview_image, 
                         (region['left'], region['top']),
                         (region['right'], region['bottom']),
                         (0, 0, 255), 2)
            
            if output_path is None:
                output_path = Path(image_path).parent / f"preview_{Path(image_path).name}"
            cv2.imwrite(str(output_path), preview_image)
        else:
            # 通常モード：マスクを適用（黒く塗りつぶす）
            masked_image = image.copy()
            masked_image[mask > 0] = 0  # マスク領域を黒に
            
            if output_path is None:
                output_path = Path(image_path).parent / f"masked_{Path(image_path).name}"
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
            print("プレビューモード: マスク範囲を赤い枠で表示します")
        
        success_count = 0
        failed_files = []
        
        for image_path in tqdm(image_files, desc="処理中"):
            # サブディレクトリ構造を保持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            if self.mask_face_center(image_path, output_file, preview):
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
    parser = argparse.ArgumentParser(description='目と鼻の部分を黒く塗りつぶす')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：マスク範囲を赤い枠で表示')
    
    args = parser.parse_args()
    
    masker = FocusedFaceMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_center(input_path, output_path, args.preview):
            mode = "プレビュー" if args.preview else "マスク"
            print(f"{mode}処理完了: {output_path or f'{"preview" if args.preview else "masked"}_{input_path.name}'}")
        else:
            print("処理に失敗しました")
    
    elif input_path.is_dir():
        # ディレクトリの処理
        suffix = '_preview' if args.preview else '_masked'
        output_dir = args.output if args.output else str(input_path) + suffix
        masker.process_directory(input_path, output_dir, args.preview)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == '__main__':
    main()