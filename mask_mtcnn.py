#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MTCNN を使用したT字型マスク（目元・鼻・口）適用スクリプト
高精度な顔検出で確実に眉毛、目、鼻、口をカバー
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
from tqdm import tqdm
import argparse


class FaceTMaskerMTCNN:
    """MTCNNを使用したT字型マスク適用クラス"""
    
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_face_landmarks(self, image):
        """MTCNNで顔とランドマークを検出"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(rgb_image)
        
        if not result:
            return None
            
        # 最も信頼度の高い顔を選択
        face = max(result, key=lambda x: x['confidence'])
        
        if face['confidence'] < 0.9:  # 信頼度90%未満はスキップ
            return None
            
        return face
    
    def create_precise_t_mask(self, image, face_data):
        """ランドマークに基づいて確実なT字マスクを作成"""
        keypoints = face_data['keypoints']
        img_h, img_w = image.shape[:2]
        
        # ランドマーク取得
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        nose = keypoints['nose']
        mouth_left = keypoints['mouth_left']
        mouth_right = keypoints['mouth_right']
        
        # 顔の特徴点に基づいた精密なT字マスク計算
        img_h, img_w = image.shape[:2]
        
        # 目の距離に基づくスケール計算
        eye_distance = abs(right_eye[0] - left_eye[0])
        
        # 眉毛の推定位置（目より上に目の距離の30%）
        eyebrow_offset = int(eye_distance * 0.3)
        left_eyebrow_y = left_eye[1] - eyebrow_offset
        right_eyebrow_y = right_eye[1] - eyebrow_offset
        
        # T字マスクの横棒（眉毛と目を確実にカバー）
        horizontal_top = max(0, min(left_eyebrow_y, right_eyebrow_y) - 25)  # 眉毛の上
        horizontal_bottom = max(left_eye[1], right_eye[1]) + 35            # 目の下
        horizontal_left = min(left_eye[0], right_eye[0]) - int(eye_distance * 0.3)  # 目の幅の30%左
        horizontal_right = max(left_eye[0], right_eye[0]) + int(eye_distance * 0.3)  # 目の幅の30%右
        
        # T字マスクの縦棒（鼻と口を確実にカバー）
        mouth_bottom = max(mouth_left[1], mouth_right[1])
        
        # 縦棒のサイズを目の距離に基づいて計算
        vertical_width = max(int(eye_distance * 0.6), 100)  # 目の距離の60%、最小100px
        
        # 縦棒の位置（目の下から口の下まで十分にカバー）
        vertical_top = min(left_eye[1], right_eye[1]) - 10   # 目の少し上から
        vertical_bottom = mouth_bottom + 40                  # 口の下40px下まで
        
        vertical_center_x = nose[0]  # 鼻の中心
        vertical_left = vertical_center_x - vertical_width // 2
        vertical_right = vertical_center_x + vertical_width // 2
        
        # 画像境界内に調整
        horizontal_top = max(0, horizontal_top)
        horizontal_bottom = min(img_h, horizontal_bottom)
        horizontal_left = max(0, horizontal_left)
        horizontal_right = min(img_w, horizontal_right)
        
        vertical_top = max(0, vertical_top)
        vertical_bottom = min(img_h, vertical_bottom)
        vertical_left = max(0, vertical_left)
        vertical_right = min(img_w, vertical_right)
        
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
            },
            'landmarks': keypoints
        }
    
    def apply_t_mask(self, image, mask_regions, preview=False):
        """T字マスクを適用"""
        if preview:
            # プレビューモード：ランドマークと枠を表示
            preview_img = image.copy()
            
            # ランドマーク描画
            landmarks = mask_regions['landmarks']
            for name, point in landmarks.items():
                cv2.circle(preview_img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
                cv2.putText(preview_img, name, (int(point[0]) + 5, int(point[1]) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
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
            
            cv2.putText(preview_img, "MTCNN Precise T-mask", (10, 30), 
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
            return False
        
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
        mode = "プレビュー" if preview else "MTCNN精密T字マスク"
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
        
        print(f"\\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        if failed_files:
            print(f"\\n失敗したファイル（最初の10個）:")
            for f in failed_files[:10]:
                print(f"  - {f}")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='MTCNN精密T字マスク適用')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：ランドマークと範囲を表示')
    
    args = parser.parse_args()
    
    masker = FaceTMaskerMTCNN()
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
        suffix = '_preview' if args.preview else '_mtcnn_masked'
        output_dir = args.output if args.output else str(input_path) + suffix
        masker.process_directory(input_path, output_dir, args.preview)
    
    else:
        print(f"エラー: {input_path} は存在しません")


if __name__ == "__main__":
    main()