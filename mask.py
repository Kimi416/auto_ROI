#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
T字型マスク（目元・鼻・口）を適用するスクリプト
医療画像の匿名化に使用。額、頬、顎は保持します。
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class FaceTMasker:
    """T字型マスク（目元・鼻・口）を適用するクラス"""
    
    def __init__(self):
        # dlib顔検出器と68点ランドマーク予測器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    def detect_landmarks(self, image):
        """顔のランドマークを検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # 最初の顔のランドマークを取得
        face = faces[0]
        shape = self.predictor(gray, face)
        points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        return points
    
    def create_t_mask(self, image_shape, landmarks):
        """T字型マスク（目元・鼻・口）を作成"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 目元マスク（両眉毛を含む広範囲）
        eye_points = []
        # 左眉毛の上部
        eye_points.extend([(p[0], p[1] - 15) for p in landmarks[17:22]])
        # 右眉毛の上部
        eye_points.extend([(p[0], p[1] - 15) for p in landmarks[22:27]])
        # 右目の下
        eye_points.extend([landmarks[28], landmarks[29], landmarks[30]])
        # 鼻の付け根まで
        eye_points.extend([landmarks[33], landmarks[32], landmarks[31]])
        # 左目に戻る
        eye_points.extend([(p[0], p[1] + 10) for p in landmarks[39:36:-1]])
        
        cv2.fillPoly(mask, [np.array(eye_points, dtype=np.int32)], 255)
        
        # 鼻マスク（広めに）
        nose_points = []
        nose_points.append((landmarks[27][0] - 20, landmarks[27][1]))
        nose_points.extend(landmarks[31:36])
        nose_points.append((landmarks[35][0], landmarks[35][1] + 10))
        nose_points.append((landmarks[31][0], landmarks[31][1] + 10))
        nose_points.append((landmarks[27][0] + 20, landmarks[27][1]))
        
        cv2.fillPoly(mask, [np.array(nose_points, dtype=np.int32)], 255)
        
        # 口マスク（唇周辺）
        mouth_points = landmarks[48:68]
        mouth_hull = cv2.convexHull(np.array(mouth_points))
        # 少し拡大
        center = np.mean(mouth_hull, axis=0)
        expanded_mouth = center + 1.2 * (mouth_hull - center)
        cv2.fillPoly(mask, [expanded_mouth.astype(np.int32)], 255)
        
        return mask
    
    
    def mask_face_t_shape(self, image_path, output_path=None):
        """画像にT字型マスク（目元・鼻・口）を適用"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # ランドマーク検出
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            print(f"顔を検出できませんでした: {image_path}")
            return False
        
        # T字型マスクを作成
        mask = self.create_t_mask(image.shape, landmarks)
        
        # マスクを適用（黒く塗りつぶす）
        masked_image = image.copy()
        masked_image[mask > 0] = [0, 0, 0]  # マスク領域を黒に
        
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
        
        for image_path in tqdm(image_files, desc="T字型マスク適用中"):
            output_file = output_path / image_path.name
            if self.mask_face_t_shape(image_path, output_file):
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
    parser = argparse.ArgumentParser(description='T字型マスク（目元・鼻・口）を適用')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    
    args = parser.parse_args()
    
    masker = FaceTMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_t_shape(input_path, output_path):
            print(f"T字型マスク処理完了: {output_path or f'masked_{input_path.name}'}")
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