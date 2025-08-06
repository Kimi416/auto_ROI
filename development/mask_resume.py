#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
T字型マスク（目元・鼻・口）を適用するスクリプト（レジューム対応版）
医療画像の匿名化に使用。額、頬、顎は保持します。
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import time
import sys
import signal


class FaceTMaskerResume:
    """T字型マスク（目元・鼻・口）を適用するクラス（レジューム対応）"""
    
    def __init__(self):
        # dlib顔検出器と68点ランドマーク予測器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.stop_processing = False
        
        # シグナルハンドラーを設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        print("\n処理を停止しています...")
        self.stop_processing = True
    
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
        nose_points.append((landmarks[27][0] - 20, landmarks[27][1]))
        
        cv2.fillPoly(mask, [np.array(nose_points, dtype=np.int32)], 255)
        
        # 口マスク（広めに）
        mouth_points = landmarks[48:68]
        mouth_points.extend([(p[0], p[1] + 15) for p in landmarks[54:60]])
        mouth_points.extend([(p[0], p[1] + 15) for p in landmarks[48:54][::-1]])
        
        cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], 255)
        
        return mask
    
    def mask_face_t_shape(self, image_path, output_path):
        """画像にT字型マスクを適用"""
        try:
            # 画像を読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"画像を読み込めませんでした: {image_path}")
                return False
            
            # 顔のランドマークを検出
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
            
        except Exception as e:
            print(f"エラーが発生しました {image_path}: {e}")
            return False
    
    def get_processed_files(self, output_path):
        """既に処理済みのファイルリストを取得"""
        processed = set()
        if output_path.exists():
            for file in output_path.rglob("*.jpg"):
                # 相対パスを保存
                rel_path = file.relative_to(output_path)
                processed.add(str(rel_path))
        return processed
    
    def process_directory(self, input_dir, output_dir):
        """ディレクトリ内のすべての画像を処理（レジューム対応）"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 画像ファイルを収集（サブディレクトリも含む）
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP', '*.tif', '*.TIF']:
            image_files.extend(input_path.rglob(ext))
        
        # 既に処理済みのファイルを取得
        processed_files = self.get_processed_files(output_path)
        
        # 未処理のファイルをフィルタリング
        remaining_files = []
        for image_path in image_files:
            relative_path = image_path.relative_to(input_path)
            if str(relative_path) not in processed_files:
                remaining_files.append(image_path)
        
        total_files = len(image_files)
        processed_count = len(processed_files)
        remaining_count = len(remaining_files)
        
        print(f"全体: {total_files}枚")
        print(f"処理済み: {processed_count}枚")
        print(f"残り: {remaining_count}枚")
        
        if remaining_count == 0:
            print("すべての画像が処理済みです。")
            return processed_count, []
        
        success_count = processed_count
        failed_files = []
        
        # プログレスバーを設定
        progress_bar = tqdm(remaining_files, desc="T字型マスク適用中", initial=0, total=remaining_count)
        
        for image_path in progress_bar:
            if self.stop_processing:
                print("\n処理が中断されました。")
                break
                
            # 相対パスを保持してサブディレクトリ構造を維持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.mask_face_t_shape(image_path, output_file):
                success_count += 1
            else:
                failed_files.append(str(image_path))
            
            # 定期的に進捗を更新
            progress_bar.set_postfix({
                'success': success_count,
                'failed': len(failed_files),
                'total': total_files
            })
        
        progress_bar.close()
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{total_files}枚")
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
    parser = argparse.ArgumentParser(description='T字型マスク（目元・鼻・口）を適用（レジューム対応）')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス', default='organized_masked_complete')
    
    args = parser.parse_args()
    
    masker = FaceTMaskerResume()
    
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
        output_dir = args.output
        masker.process_directory(input_path, output_dir)
    
    else:
        print(f"指定されたパスが見つかりません: {input_path}")


if __name__ == "__main__":
    main()