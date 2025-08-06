#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顔の中心部をT字型にマスクするスクリプト（高度な顔ランドマーク検出版）
dlibとface_recognitionを使用して正確に眉毛と目を検出
額、頬、顎は残し、目・鼻・口の部分をT字型に黒く塗りつぶす
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


class AdvancedTShapeFaceMasker:
    """高度な顔ランドマーク検出を使用したT字型マスククラス"""
    
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
        # 眉毛の最上部から大きな余裕を持って上に（確実に覆うため）
        # 横顔の場合はより大きな余裕を取る
        face_height = np.max(jaw[:, 1]) - np.min(all_eyebrows[:, 1])
        extra_margin = max(100, int(face_height * 0.15))  # 顔の高さの15%または最低100ピクセル
        eyebrow_top = np.min(all_eyebrows[:, 1]) - extra_margin
        # 目の最下部から余裕を持って下に
        eye_bottom = np.max(all_eyes[:, 1]) + 15
        
        # 横棒の左右：顔の幅から少し内側
        face_left = np.min(jaw[:, 0])
        face_right = np.max(jaw[:, 0])
        face_width = face_right - face_left
        
        horizontal_left = face_left + int(face_width * 0.05)  # 5%内側
        horizontal_right = face_right - int(face_width * 0.05)  # 5%内側
        
        # T字型マスクの縦棒（鼻から口の領域）
        # 眉毛と目の間から開始
        vertical_top = np.min(all_eyebrows[:, 1]) + 10
        # 口の下端まで
        vertical_bottom = np.max(outer_lip[:, 1]) + 10
        
        # 縦棒の左右：鼻を中心に適切な幅
        nose_center_x = np.mean(nose_tip[:, 0])
        nose_width = np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0])
        
        # 鼻の幅の2倍程度を縦棒の幅とする（より細く）
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
            },
            'landmarks': landmarks  # デバッグ用
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
        regions = self.detect_face_features_advanced(image)
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
            
            # ランドマークポイントを表示（デバッグ用）
            if 'landmarks' in regions:
                landmarks = regions['landmarks']
                # 眉毛のポイントを緑で表示
                for idx in self.LANDMARKS['left_eyebrow'] + self.LANDMARKS['right_eyebrow']:
                    cv2.circle(preview_image, (landmarks[idx][0], landmarks[idx][1]), 2, (0, 255, 0), -1)
                # 目のポイントを青で表示
                for idx in self.LANDMARKS['left_eye'] + self.LANDMARKS['right_eye']:
                    cv2.circle(preview_image, (landmarks[idx][0], landmarks[idx][1]), 2, (255, 0, 0), -1)
            
            # T字の説明テキストを追加
            cv2.putText(preview_image, "Advanced T-shape mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if output_path is None:
                output_path = Path(image_path).parent / f"preview_adv_{Path(image_path).name}"
            cv2.imwrite(str(output_path), preview_image)
        else:
            # 通常モード：マスクを適用（黒く塗りつぶす）
            mask = self.create_t_shape_mask(image.shape, regions)
            masked_image = image.copy()
            masked_image[mask > 0] = 0  # マスク領域を黒に
            
            if output_path is None:
                output_path = Path(image_path).parent / f"masked_adv_{Path(image_path).name}"
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
            print("プレビューモード: 高度なT字型マスク範囲を赤い枠で表示します")
        else:
            print("高度なT字型マスクモード: 正確な眉毛・目の検出で額・頬・顎を残します")
        
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
    parser = argparse.ArgumentParser(description='高度な顔検出で顔の中心部をT字型に黒く塗りつぶす')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：T字型マスク範囲とランドマークを表示')
    
    args = parser.parse_args()
    
    masker = AdvancedTShapeFaceMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_t_shape(input_path, output_path, args.preview):
            mode = "プレビュー" if args.preview else "マスク"
            print(f"{mode}処理完了: {output_path or f'{"preview_adv" if args.preview else "masked_adv"}_{input_path.name}'}")
        else:
            print("処理に失敗しました")
    
    elif input_path.is_dir():
        # ディレクトリの処理
        suffix = '_adv_preview' if args.preview else '_adv_masked'
        output_dir = args.output if args.output else str(input_path) + suffix
        masker.process_directory(input_path, output_dir, args.preview)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == '__main__':
    main()