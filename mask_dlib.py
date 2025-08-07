#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dlib を使用したT字型マスク（目元・鼻・口）適用スクリプト
高精度な顔検出とランドマーク検出で確実に眉毛、目、鼻、口をカバー
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from tqdm import tqdm
import argparse


class FaceTMaskerDlib:
    """dlibを使用したT字型マスク適用クラス"""
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # 顔ランドマーク検出器（68点）
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.use_landmarks = True
        except RuntimeError:
            print("ランドマーク検出器が見つかりません。基本的な顔検出のみ使用します。")
            self.use_landmarks = False
    
    def detect_face_landmarks(self, image):
        """dlibで顔とランドマークを検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        # 最大の顔を選択
        face = max(faces, key=lambda f: f.width() * f.height())
        
        result = {
            'face_rect': (face.left(), face.top(), face.width(), face.height())
        }
        
        if self.use_landmarks:
            landmarks = self.predictor(gray, face)
            
            # 重要な特徴点を抽出
            result['landmarks'] = {
                'left_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)],
                'right_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)],
                'left_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
                'right_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
                'nose': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)],
                'mouth': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            }
        
        return result
    
    def create_precise_t_mask(self, image, face_data):
        """ランドマークに基づいて確実なT字マスクを作成"""
        img_h, img_w = image.shape[:2]
        
        if self.use_landmarks and 'landmarks' in face_data:
            landmarks = face_data['landmarks']
            
            # 眉毛の範囲
            left_eyebrow = landmarks['left_eyebrow']
            right_eyebrow = landmarks['right_eyebrow']
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            mouth = landmarks['mouth']
            
            # 眉毛の上端と下端
            eyebrow_top = min([p[1] for p in left_eyebrow + right_eyebrow]) - 20
            eye_bottom = max([p[1] for p in left_eye + right_eye]) + 20
            
            # 目の左右端
            eye_left = min([p[0] for p in left_eye + right_eye]) - 30
            eye_right = max([p[0] for p in left_eye + right_eye]) + 30
            
            # 口の範囲
            mouth_top = min([p[1] for p in mouth]) - 10
            mouth_bottom = max([p[1] for p in mouth]) + 30
            mouth_left = min([p[0] for p in mouth]) - 20
            mouth_right = max([p[0] for p in mouth]) + 20
            
            # 鼻の中心
            nose_center_x = sum([p[0] for p in landmarks['nose']]) // len(landmarks['nose'])
            
        else:
            # ランドマークが使用できない場合、顔の位置から推定
            x, y, w, h = face_data['face_rect']
            
            # 眉毛と目の範囲（顔の上部）
            eyebrow_top = y + int(h * 0.15)
            eye_bottom = y + int(h * 0.5)
            eye_left = x + int(w * 0.1)
            eye_right = x + int(w * 0.9)
            
            # 口の範囲（顔の下部）
            mouth_top = y + int(h * 0.6)
            mouth_bottom = y + int(h * 0.9)
            mouth_left = x + int(w * 0.2)
            mouth_right = x + int(w * 0.8)
            
            # 鼻の中心
            nose_center_x = x + w // 2
        
        # T字マスクの横棒（眉毛と目をカバー）
        horizontal_top = max(0, eyebrow_top)
        horizontal_bottom = min(img_h, eye_bottom)
        horizontal_left = max(0, eye_left)
        horizontal_right = min(img_w, eye_right)
        
        # T字マスクの縦棒（鼻と口をカバー）
        vertical_top = max(0, eye_bottom - 10)  # 目の少し上から
        vertical_bottom = min(img_h, mouth_bottom)
        
        # 縦棒の幅を計算
        vertical_width = max(mouth_right - mouth_left, 80)
        vertical_left = max(0, nose_center_x - vertical_width // 2)
        vertical_right = min(img_w, nose_center_x + vertical_width // 2)
        
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
    
    def apply_t_mask(self, image, mask_regions, preview=False):
        """T字マスクを適用"""
        if preview:
            # プレビューモード：ランドマークと枠を表示
            preview_img = image.copy()
            
            # T字マスク範囲を赤い枠で表示
            h_region = mask_regions['horizontal']
            v_region = mask_regions['vertical']
            
            # 横棒
            cv2.rectangle(preview_img, 
                         (h_region['left'], h_region['top']),
                         (h_region['right'], h_region['bottom']),
                         (0, 0, 255), 2)
            
            # 縦棒
            cv2.rectangle(preview_img, 
                         (v_region['left'], v_region['top']),
                         (v_region['right'], v_region['bottom']),
                         (0, 0, 255), 2)
            
            cv2.putText(preview_img, "Dlib Precise T-mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return preview_img
        else:
            # 実際のマスク適用
            masked_img = image.copy()
            h_region = mask_regions['horizontal']
            v_region = mask_regions['vertical']
            
            # 横棒（眉毛と目）を黒で塗りつぶし
            masked_img[h_region['top']:h_region['bottom'], 
                      h_region['left']:h_region['right']] = [0, 0, 0]
            
            # 縦棒（鼻と口）を黒で塗りつぶし
            masked_img[v_region['top']:v_region['bottom'], 
                      v_region['left']:v_region['right']] = [0, 0, 0]
            
            return masked_img
    
    def process_image(self, image_path, output_path=None, preview=False):
        """単一画像を処理"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # 顔検出
        face_data = self.detect_face_landmarks(image)
        if face_data is None:
            print(f"顔が検出できませんでした: {image_path}")
            # 体幹部画像など顔がない画像は元画像をそのままコピー
            if output_path is None:
                prefix = "preview_" if preview else "masked_"
                output_path = Path(image_path).parent / f"{prefix}{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
            return True
        
        # T字マスク領域計算
        mask_regions = self.create_precise_t_mask(image, face_data)
        
        # マスク適用
        result_image = self.apply_t_mask(image, mask_regions, preview)
        
        # 保存
        if output_path is None:
            prefix = "preview_" if preview else "masked_"
            output_path = Path(image_path).parent / f"{prefix}{Path(image_path).name}"
        
        cv2.imwrite(str(output_path), result_image)
        return True
    
    def process_directory(self, input_dir, output_dir, preview=False):
        """ディレクトリ内の全画像を処理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 画像ファイルを収集
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP']:
            image_files.extend(input_path.rglob(ext))
        
        print(f"処理対象: {len(image_files)}枚の画像")
        mode = "プレビュー" if preview else "dlib精密T字マスク"
        print(f"モード: {mode}")
        
        success_count = 0
        failed_files = []
        
        for image_path in tqdm(image_files, desc="処理中"):
            # サブディレクトリ構造を保持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            if self.process_image(image_path, output_file, preview):
                success_count += 1
            else:
                failed_files.append(str(image_path))
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        if failed_files:
            print(f"\n失敗したファイル（最初の10個）:")
            for f in failed_files[:10]:
                print(f"  - {f}")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='dlib精密T字マスク適用')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：ランドマークと範囲を表示')
    
    args = parser.parse_args()
    
    masker = FaceTMaskerDlib()
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.process_image(input_path, output_path, args.preview):
            mode = "プレビュー" if args.preview else "マスク"
            print(f"{mode}処理完了: {output_path or f'processed_{input_path.name}'}")
        else:
            print("処理に失敗しました")
    
    elif input_path.is_dir():
        # ディレクトリの処理
        suffix = '_preview' if args.preview else '_dlib_masked'
        output_dir = args.output if args.output else str(input_path) + suffix
        masker.process_directory(input_path, output_dir, args.preview)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == "__main__":
    main()