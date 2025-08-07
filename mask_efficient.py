#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
効率的なT字マスク処理 - 既処理済み画像をスキップ
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import urllib.request
import bz2
import os


class EfficientTShapeFaceMasker:
    """効率的なT字型マスククラス - 既処理済み画像をスキップ"""
    
    def __init__(self):
        # dlibの顔検出器を初期化
        self.face_detector = dlib.get_frontal_face_detector()
        
        # 68点ランドマークモデルのパスを設定
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        # モデルファイルが存在しない場合はダウンロード
        if not os.path.exists(self.predictor_path):
            self.download_predictor_model()
        
        # ランドマーク予測器を初期化
        self.predictor = dlib.shape_predictor(self.predictor_path)
        
        # 顔の各部位のランドマークインデックス
        self.LANDMARKS = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lip': list(range(48, 60)),
            'inner_lip': list(range(60, 68))
        }
    
    def download_predictor_model(self):
        """68点ランドマークモデルをダウンロード"""
        print("顔ランドマークモデルをダウンロードしています...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = "shape_predictor_68_face_landmarks.dat.bz2"
        
        # ダウンロード
        urllib.request.urlretrieve(url, compressed_path)
        
        # 解凍
        with open(compressed_path, 'rb') as f:
            decompressed_data = bz2.decompress(f.read())
        
        with open(self.predictor_path, 'wb') as f:
            f.write(decompressed_data)
        
        # 圧縮ファイルを削除
        os.remove(compressed_path)
        print("モデルのダウンロードが完了しました")
    
    def get_landmarks(self, image):
        """画像から顔のランドマークを取得"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 顔を検出
        faces = self.face_detector(gray, 1)
        
        if len(faces) == 0:
            return None
        
        # 最初の顔のランドマークを取得
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # ランドマークポイントを配列に変換
        points = []
        for i in range(68):
            point = landmarks.part(i)
            points.append([point.x, point.y])
        
        return np.array(points)
    
    def detect_face_features_advanced(self, image):
        """高度な方法で顔の特徴を検出してT字型マスクの領域を計算"""
        landmarks = self.get_landmarks(image)
        
        if landmarks is None:
            return None
        
        # 眉毛の正確な位置を取得
        left_eyebrow = landmarks[self.LANDMARKS['left_eyebrow']]
        right_eyebrow = landmarks[self.LANDMARKS['right_eyebrow']]
        all_eyebrows = np.vstack([left_eyebrow, right_eyebrow])
        
        # 目の正確な位置を取得
        left_eye = landmarks[self.LANDMARKS['left_eye']]
        right_eye = landmarks[self.LANDMARKS['right_eye']]
        all_eyes = np.vstack([left_eye, right_eye])
        
        # 鼻の位置を取得
        nose_bridge = landmarks[self.LANDMARKS['nose_bridge']]
        nose_tip = landmarks[self.LANDMARKS['nose_tip']]
        
        # 口の位置を取得
        outer_lip = landmarks[self.LANDMARKS['outer_lip']]
        
        # 顎のラインを取得
        jaw = landmarks[self.LANDMARKS['jaw']]
        
        # T字型マスクの横棒（眉毛と目を含む領域）
        face_height = np.max(jaw[:, 1]) - np.min(all_eyebrows[:, 1])
        extra_margin = max(100, int(face_height * 0.15))  # 顔の高さの15%または最低100ピクセル
        eyebrow_top = np.min(all_eyebrows[:, 1]) - extra_margin
        eye_bottom = np.max(all_eyes[:, 1]) + 15
        
        # 横棒の左右：顔の幅から少し内側
        face_left = np.min(jaw[:, 0])
        face_right = np.max(jaw[:, 0])
        face_width = face_right - face_left
        
        horizontal_left = face_left + int(face_width * 0.05)  # 5%内側
        horizontal_right = face_right - int(face_width * 0.05)  # 5%内側
        
        # T字型マスクの縦棒（鼻から口の領域）
        vertical_top = np.min(all_eyebrows[:, 1]) + 10
        vertical_bottom = np.max(outer_lip[:, 1]) + 10
        
        # 縦棒の左右：鼻を中心に適切な幅
        nose_center_x = np.mean(nose_tip[:, 0])
        nose_width = np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0])
        
        # 鼻の幅の2倍程度を縦棒の幅とする
        vertical_width = nose_width * 2
        vertical_left = int(nose_center_x - vertical_width / 2)
        vertical_right = int(nose_center_x + vertical_width / 2)
        
        return {
            'horizontal': {
                'top': int(eyebrow_top),
                'bottom': int(eye_bottom),
                'left': int(horizontal_left),
                'right': int(horizontal_right)
            },
            'vertical': {
                'top': int(vertical_top),
                'bottom': int(vertical_bottom),
                'left': int(vertical_left),
                'right': int(vertical_right)
            }
        }
    
    def mask_face_t_shape(self, image_path, output_path):
        """画像の顔中心部をT字型にマスク"""
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # 顔の特徴を検出
        regions = self.detect_face_features_advanced(image)
        if regions is None:
            return False
        
        # マスクを適用（黒く塗りつぶす）
        masked_image = image.copy()
        
        # 横棒を描画
        horizontal = regions['horizontal']
        cv2.rectangle(masked_image, 
                     (horizontal['left'], horizontal['top']),
                     (horizontal['right'], horizontal['bottom']),
                     (0, 0, 0), -1)
        
        # 縦棒を描画
        vertical = regions['vertical']
        cv2.rectangle(masked_image, 
                     (vertical['left'], vertical['top']),
                     (vertical['right'], vertical['bottom']),
                     (0, 0, 0), -1)
        
        cv2.imwrite(str(output_path), masked_image)
        return True
    
    def process_directory_efficient(self, input_dir, output_dir):
        """既処理済み画像をスキップして効率的に処理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # すべての入力画像を収集
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
            image_files.extend(input_path.rglob(ext))
        
        # 既処理済み画像を確認
        processed_files = set()
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
            processed_files.update(output_path.rglob(ext))
        
        # 処理済みファイル名のセットを作成
        processed_names = {f.name for f in processed_files}
        
        # 未処理のファイルのみを抽出
        unprocessed_files = []
        for img_file in image_files:
            if img_file.name not in processed_names:
                unprocessed_files.append(img_file)
        
        print(f"総画像数: {len(image_files)}枚")
        print(f"処理済み: {len(processed_names)}枚")
        print(f"未処理: {len(unprocessed_files)}枚")
        
        if len(unprocessed_files) == 0:
            print("すべての画像が処理済みです")
            return len(processed_names), []
        
        print("高度なT字型マスクモード: 正確な眉毛・目の検出で額・頬・顎を残します")
        
        success_count = len(processed_names)  # 既処理済み分を含む
        failed_files = []
        
        for image_path in tqdm(unprocessed_files, desc="未処理画像を処理中"):
            # サブディレクトリ構造を保持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            if self.mask_face_t_shape(image_path, output_file):
                success_count += 1
            else:
                failed_files.append(str(image_path))
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='効率的T字マスク処理（既処理済み画像スキップ）')
    parser.add_argument('input', help='入力ディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス')
    
    args = parser.parse_args()
    
    masker = EfficientTShapeFaceMasker()
    
    input_path = Path(args.input)
    output_dir = args.output if args.output else 'organized_masked'
    
    if input_path.is_dir():
        masker.process_directory_efficient(input_path, output_dir)
    else:
        print(f"エラー: {input_path} はディレクトリではありません")


if __name__ == '__main__':
    main()