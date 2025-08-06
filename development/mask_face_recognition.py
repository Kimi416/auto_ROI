#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顔の中央部分（眉毛の上から目、鼻、口を含む領域）を黒く塗りつぶすスクリプト
face_recognitionライブラリを使用
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class FaceCenterMasker:
    """顔の中央部分をマスクするクラス"""
    
    def __init__(self):
        pass
    
    def detect_face_landmarks(self, image):
        """顔のランドマークを検出"""
        # face_recognitionはRGB画像を期待
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔のランドマークを検出
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        
        if not face_landmarks_list:
            return None
        
        # 最初の顔のランドマークを返す
        return face_landmarks_list[0]
    
    def create_center_mask(self, image_shape, landmarks):
        """顔の中央部分をマスク"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not landmarks:
            return mask
        
        # 眉毛の上部を取得
        left_eyebrow = landmarks.get('left_eyebrow', [])
        right_eyebrow = landmarks.get('right_eyebrow', [])
        
        # 口の下部を取得
        bottom_lip = landmarks.get('bottom_lip', [])
        
        if left_eyebrow and right_eyebrow and bottom_lip:
            # 眉毛の最上部のy座標を見つける
            eyebrow_points = left_eyebrow + right_eyebrow
            eyebrow_top_y = min([point[1] for point in eyebrow_points]) - 20  # マージンを追加
            
            # 口の最下部のy座標を見つける
            mouth_bottom_y = max([point[1] for point in bottom_lip]) + 10  # マージンを追加
            
            # 左右の境界を決定（顔の輪郭から）
            chin = landmarks.get('chin', [])
            if chin:
                # 顔の幅を推定
                chin_xs = [point[0] for point in chin]
                face_left = min(chin_xs)
                face_right = max(chin_xs)
                
                # 顔の中央部分の幅を計算（顔幅の60%程度）
                face_width = face_right - face_left
                center_width = int(face_width * 0.6)
                face_center_x = (face_left + face_right) // 2
                
                left_boundary = face_center_x - center_width // 2
                right_boundary = face_center_x + center_width // 2
            else:
                # 代替方法：目の位置から推定
                left_eye = landmarks.get('left_eye', [])
                right_eye = landmarks.get('right_eye', [])
                
                if left_eye and right_eye:
                    left_eye_center = np.mean(left_eye, axis=0)
                    right_eye_center = np.mean(right_eye, axis=0)
                    
                    left_boundary = int(left_eye_center[0]) - 30
                    right_boundary = int(right_eye_center[0]) + 30
                else:
                    return mask
            
            # 矩形を描画
            cv2.rectangle(mask, 
                         (left_boundary, eyebrow_top_y),
                         (right_boundary, mouth_bottom_y),
                         255, -1)
        
        return mask
    
    def mask_face_center(self, image_path, output_path=None):
        """画像の顔中央部分をマスク"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # ランドマーク検出
        landmarks = self.detect_face_landmarks(image)
        if landmarks is None:
            print(f"顔を検出できませんでした: {image_path}")
            return False
        
        # マスクを作成
        mask = self.create_center_mask(image.shape, landmarks)
        
        # マスクを適用（黒く塗りつぶす）
        masked_image = image.copy()
        masked_image[mask > 0] = 0  # マスク領域を黒に
        
        # 保存
        if output_path is None:
            output_path = Path(image_path).parent / f"masked_{Path(image_path).name}"
        
        cv2.imwrite(str(output_path), masked_image)
        return True
    
    def process_directory(self, input_dir, output_dir):
        """ディレクトリ内のすべての画像を処理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 画像ファイルを収集
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            image_files.extend(input_path.glob(ext))
        
        print(f"処理対象: {len(image_files)}枚の画像")
        
        success_count = 0
        failed_files = []
        
        for image_path in tqdm(image_files, desc="処理中"):
            output_file = output_path / image_path.name
            if self.mask_face_center(image_path, output_file):
                success_count += 1
            else:
                failed_files.append(str(image_path))
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        if failed_files:
            # 失敗したファイルをJSONで保存
            with open(output_path / 'failed_files.json', 'w') as f:
                json.dump(failed_files, f, indent=2)
            
            print(f"\n失敗したファイル（最初の10個）:")
            for f in failed_files[:10]:
                print(f"  - {f}")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='顔の中央部分を黒く塗りつぶす')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    
    args = parser.parse_args()
    
    masker = FaceCenterMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_center(input_path, output_path):
            print(f"マスク処理完了: {output_path or f'masked_{input_path.name}'}")
        else:
            print("マスク処理に失敗しました")
    
    elif input_path.is_dir():
        # ディレクトリの処理
        output_dir = args.output if args.output else input_path / 'masked_output'
        masker.process_directory(input_path, output_dir)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == '__main__':
    main()