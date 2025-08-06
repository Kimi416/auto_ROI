#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
並列T字マスク処理 - 高速処理版
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from tqdm import tqdm
import argparse
import urllib.request
import bz2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class ParallelTShapeFaceMasker:
    """並列T字型マスククラス"""
    
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


def process_single_image(args):
    """単一画像の処理（並列処理用）"""
    image_path, output_path, predictor_path = args
    
    try:
        # 各プロセスで独自のインスタンスを作成
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        # ランドマーク定義
        LANDMARKS = {
            'left_eyebrow': list(range(22, 27)),
            'right_eyebrow': list(range(17, 22)),
            'left_eye': list(range(42, 48)),
            'right_eye': list(range(36, 42)),
            'nose_tip': list(range(31, 36)),
            'outer_lip': list(range(48, 60)),
            'jaw': list(range(0, 17))
        }
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # 顔検出
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        
        if len(faces) == 0:
            return False
        
        # ランドマーク取得
        face = faces[0]
        landmarks = predictor(gray, face)
        
        # ランドマークポイントを配列に変換
        points = []
        for i in range(68):
            point = landmarks.part(i)
            points.append([point.x, point.y])
        landmarks_array = np.array(points)
        
        # T字型マスク領域計算
        left_eyebrow = landmarks_array[LANDMARKS['left_eyebrow']]
        right_eyebrow = landmarks_array[LANDMARKS['right_eyebrow']]
        all_eyebrows = np.vstack([left_eyebrow, right_eyebrow])
        
        left_eye = landmarks_array[LANDMARKS['left_eye']]
        right_eye = landmarks_array[LANDMARKS['right_eye']]
        all_eyes = np.vstack([left_eye, right_eye])
        
        nose_tip = landmarks_array[LANDMARKS['nose_tip']]
        outer_lip = landmarks_array[LANDMARKS['outer_lip']]
        jaw = landmarks_array[LANDMARKS['jaw']]
        
        # 横棒領域
        face_height = np.max(jaw[:, 1]) - np.min(all_eyebrows[:, 1])
        extra_margin = max(100, int(face_height * 0.15))
        eyebrow_top = np.min(all_eyebrows[:, 1]) - extra_margin
        eye_bottom = np.max(all_eyes[:, 1]) + 15
        
        face_left = np.min(jaw[:, 0])
        face_right = np.max(jaw[:, 0])
        face_width = face_right - face_left
        horizontal_left = face_left + int(face_width * 0.05)
        horizontal_right = face_right - int(face_width * 0.05)
        
        # 縦棒領域
        vertical_top = np.min(all_eyebrows[:, 1]) + 10
        vertical_bottom = np.max(outer_lip[:, 1]) + 10
        nose_center_x = np.mean(nose_tip[:, 0])
        nose_width = np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0])
        vertical_width = nose_width * 2
        vertical_left = int(nose_center_x - vertical_width / 2)
        vertical_right = int(nose_center_x + vertical_width / 2)
        
        # マスク適用
        masked_image = image.copy()
        
        # 横棒
        cv2.rectangle(masked_image, 
                     (int(horizontal_left), int(eyebrow_top)),
                     (int(horizontal_right), int(eye_bottom)),
                     (0, 0, 0), -1)
        
        # 縦棒
        cv2.rectangle(masked_image, 
                     (int(vertical_left), int(vertical_top)),
                     (int(vertical_right), int(vertical_bottom)),
                     (0, 0, 0), -1)
        
        # 出力
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), masked_image)
        return True
        
    except Exception as e:
        print(f"エラー処理 {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='並列T字マスク処理')
    parser.add_argument('input', help='入力ディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス', default='organized_masked_parallel')
    parser.add_argument('-j', '--jobs', type=int, help='並列処理数', default=multiprocessing.cpu_count())
    
    args = parser.parse_args()
    
    # マスカーの初期化（モデルファイルのダウンロードのため）
    masker = ParallelTShapeFaceMasker()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 全画像ファイル収集
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
        image_files.extend(input_path.rglob(ext))
    
    # 既処理済み画像確認
    processed_files = set()
    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
        processed_files.update(output_path.rglob(ext))
    processed_names = {f.name for f in processed_files}
    
    # 未処理ファイル抽出
    unprocessed_files = []
    for img_file in image_files:
        if img_file.name not in processed_names:
            unprocessed_files.append(img_file)
    
    print(f"総画像数: {len(image_files)}枚")
    print(f"処理済み: {len(processed_names)}枚")
    print(f"未処理: {len(unprocessed_files)}枚")
    print(f"並列処理数: {args.jobs}")
    
    if len(unprocessed_files) == 0:
        print("すべての画像が処理済みです")
        return
    
    # 並列処理用の引数リスト作成
    process_args = []
    for image_path in unprocessed_files:
        relative_path = image_path.relative_to(input_path)
        output_file = output_path / relative_path
        process_args.append((image_path, output_file, masker.predictor_path))
    
    # 並列処理実行
    success_count = len(processed_names)
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_single_image, arg): arg for arg in process_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="並列処理中"):
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"エラー: {e}")
    
    print(f"\n処理完了:")
    print(f"- 成功: {success_count}/{len(image_files)}枚")
    print(f"- 失敗: {failed_count}枚")


if __name__ == '__main__':
    main()