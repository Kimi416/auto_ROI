#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MTCNNを使用した顔部位マスク適用スクリプト
高精度な顔ランドマーク検出で眉毛・目・鼻・口を確実にマスク
医療画像の匿名化に使用。額、頬、顎は保持します。
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from mtcnn import MTCNN


class FaceMasker:
    """MTCNNを使用した顔部位マスク適用クラス"""
    
    def __init__(self):
        try:
            self.detector = MTCNN()
            self.use_mtcnn = True
            print("MTCNNモデルをロードしました")
        except Exception as e:
            print(f"MTCNN読み込み失敗: {e}")
            # フォールバックとしてOpenCV顔検出器を使用
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.use_mtcnn = False
            print("OpenCV顔検出器を使用します")
    
    def detect_face_with_mtcnn(self, image):
        """MTCNNを使用して顔とランドマークを検出"""
        if not self.use_mtcnn:
            return None
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.detector.detect_faces(rgb_image)
            
            if not result:
                return None
                
            # 最も信頼度の高い顔を選択
            face = max(result, key=lambda x: x['confidence'])
            
            if face['confidence'] < 0.9:  # 信頼度90%未満はスキップ
                return None
                
            return face
            
        except Exception as e:
            print(f"MTCNN検出エラー: {e}")
            return None
    
    def create_precise_mask_regions(self, image):
        """MTCNNランドマークに基づいて顔部位マスク領域を作成"""
        img_h, img_w = image.shape[:2]
        
        # まずMTCNNで顔検出を試行
        mtcnn_result = self.detect_face_with_mtcnn(image)
        
        if mtcnn_result and 'keypoints' in mtcnn_result:
            # MTCNNランドマーク検出成功時
            keypoints = mtcnn_result['keypoints']
            
            # ランドマーク取得
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            nose = keypoints['nose']
            mouth_left = keypoints['mouth_left']
            mouth_right = keypoints['mouth_right']
            
            # 眉毛の推定位置（目より上に目の距離の50%）
            eye_distance = abs(right_eye[0] - left_eye[0])
            eyebrow_offset = int(eye_distance * 0.4)
            
            # Advanced T-shape mask のロジックを完全に再現
            regions = []
            
            # 顔の輪郭（顎のライン）を推定（MTCNNにはないので近似）
            # left_eye と right_eye の距離から顔の幅を推定
            eye_y = (left_eye[1] + right_eye[1]) / 2
            mouth_y = (mouth_left[1] + mouth_right[1]) / 2
            
            # 顔の高さを推定（目から口まで + さらに下）
            face_height = mouth_y - eye_y
            jaw_y = mouth_y + face_height * 0.5  # 口から下にさらに50%
            
            # 顔の左右端を推定（目の間隔の2.5倍程度）
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            face_half_width = eye_distance * 1.25
            face_left = face_center_x - face_half_width
            face_right = face_center_x + face_half_width
            face_width = face_right - face_left
            
            # 1. 横長の矩形：Advanced T-shape と完全同等の計算
            # 眉毛の最上部から大きな余裕を持って上に（確実に覆うため）
            extra_margin = max(100, int(face_height * 0.15))  # 顔の高さの15%または最低100px
            eyebrow_top = min(left_eye[1], right_eye[1]) - eyebrow_offset
            horizontal_top = max(0, int(eyebrow_top - extra_margin))
            
            # 目の最下部から余裕を持って下に
            horizontal_bottom = min(img_h, int(max(left_eye[1], right_eye[1]) + 15))
            
            # 横棒の左右：顔の幅から5%内側（Advanced T-shape と同じ）
            horizontal_left = max(0, int(face_left + face_width * 0.05))
            horizontal_right = min(img_w, int(face_right - face_width * 0.05))
            regions.append((horizontal_left, horizontal_top, horizontal_right, horizontal_bottom))
            
            # 2. 縦長の矩形：Advanced T-shape と完全同等の計算
            # 眉毛と目の間から開始
            vertical_top = max(0, int(eyebrow_top + 10))
            # 口の下端まで
            vertical_bottom = min(img_h, int(max(mouth_left[1], mouth_right[1]) + 10))
            
            # 縦棒の左右：鼻を中心に適切な幅（Advanced T-shapeより細く）
            nose_center_x = nose[0]
            # 鼻の幅を目の間隔から推定（Advanced T-shapeでは鼻の幅の2倍だが、MTCNNでは推定）
            estimated_nose_width = eye_distance * 0.15  # より細く設定
            vertical_width = estimated_nose_width * 2
            vertical_left = max(0, int(nose_center_x - vertical_width / 2))
            vertical_right = min(img_w, int(nose_center_x + vertical_width / 2))
            regions.append((vertical_left, vertical_top, vertical_right, vertical_bottom))
            
            return regions
            
        else:
            # MTCNNフォールバック：OpenCVまたは固定位置
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if hasattr(self, 'face_cascade') else []
            
            if len(faces) > 0:
                # OpenCV顔検出成功
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # 顔の中央部を大きくマスク
                mask_top = y + int(h * 0.15)
                mask_bottom = y + int(h * 0.80)
                mask_left = x + int(w * 0.20)
                mask_right = x + int(w * 0.80)
                
                return [(mask_left, mask_top, mask_right, mask_bottom)]
            else:
                # 固定位置マスク
                mask_top = int(img_h * 0.25)
                mask_bottom = int(img_h * 0.70)
                mask_left = int(img_w * 0.30)
                mask_right = int(img_w * 0.70)
                
                return [(mask_left, mask_top, mask_right, mask_bottom)]
    
    def create_face_mask(self, image_shape, regions):
        """各部位の矩形マスクを作成"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if regions is None or len(regions) == 0:
            return mask
        
        # 各部位領域を矩形でマスク
        for region in regions:
            left, top, right, bottom = region
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
        
        return mask
    
    
    def mask_face_precise(self, image_path, output_path=None, preview=False):
        """画像にMTCNN精密マスクを適用"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return False
        
        # 顔部位領域の検出
        regions = self.create_precise_mask_regions(image)
        if regions is None or len(regions) == 0:
            # 体幹部画像など顔がない画像は元画像をそのままコピー
            if output_path is None:
                output_path = Path(image_path).parent / f"{'preview_' if preview else 'masked_'}{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
            return True
        
        if preview:
            # プレビューモード：マスク範囲を赤い枠で表示
            preview_image = image.copy()
            
            # 各部位領域を赤い枠で描画
            for i, region in enumerate(regions):
                left, top, right, bottom = region
                cv2.rectangle(preview_image, (left, top), (right, bottom), (0, 0, 255), 2)
                # 部位名をラベル表示
                labels = ['Eyebrows', 'Eyes', 'Nose', 'Mouth']
                if i < len(labels):
                    cv2.putText(preview_image, labels[i], (left, top-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # タイトル追加
            cv2.putText(preview_image, "MTCNN Precise Face Mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if output_path is None:
                output_path = Path(image_path).parent / f"preview_{Path(image_path).name}"
            cv2.imwrite(str(output_path), preview_image)
        else:
            # 通常モード：マスクを適用
            masked_image = image.copy()
            
            # 各領域を直接マスク（デバッグ用にregionsの内容を確認）
            if regions and len(regions) > 0:
                print(f"マスク領域数: {len(regions)}")
                for i, region in enumerate(regions):
                    left, top, right, bottom = region
                    print(f"領域{i+1}: ({left}, {top}) to ({right}, {bottom})")
                    # 直接矩形を黒で塗りつぶし
                    cv2.rectangle(masked_image, (left, top), (right, bottom), (0, 0, 0), -1)
            else:
                print("マスク領域が見つかりません")
            
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
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.bmp', '*.BMP', '*.tif', '*.TIF']:
            image_files.extend(input_path.rglob(ext))
        
        print(f"処理対象: {len(image_files)}枚の画像")
        if preview:
            print("プレビューモード: MTCNN精密マスク範囲を赤い枠で表示")
        else:
            print("MTCNN精密マスクモード: 眉毛・目・鼻・口を個別に精密マスク")
        
        success_count = 0
        failed_files = []
        
        for image_path in tqdm(image_files, desc="処理中"):
            # サブディレクトリ構造を保持
            relative_path = image_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            if self.mask_face_precise(image_path, output_file, preview):
                success_count += 1
            else:
                failed_files.append(str(image_path))
        
        print(f"\n処理完了:")
        print(f"- 成功: {success_count}/{len(image_files)}枚")
        print(f"- 失敗: {len(failed_files)}枚")
        
        if failed_files:
            # 失敗したファイルをJSONで保存
            with open(output_path / 'failed_files.json', 'w', encoding='utf-8') as f:
                json.dump(failed_files, f, indent=2, ensure_ascii=False)
            
            print(f"\n失敗したファイル（最初の10個）:")
            for f in failed_files[:10]:
                print(f"  - {f}")
            
            if len(failed_files) > 10:
                print(f"  ... 他 {len(failed_files) - 10} ファイル")
        
        return success_count, failed_files


def main():
    parser = argparse.ArgumentParser(description='MTCNN精密顔部位マスクを適用')
    parser.add_argument('input', help='入力画像またはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力先のパス（省略時は入力と同じ場所）')
    parser.add_argument('--preview', action='store_true', 
                       help='プレビューモード：精密マスク範囲を表示')
    
    args = parser.parse_args()
    
    masker = FaceMasker()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイルの処理
        output_path = args.output if args.output else None
        if masker.mask_face_precise(input_path, output_path, args.preview):
            mode = "プレビュー" if args.preview else "マスク"
            print(f"{mode}処理完了: {output_path or f'{'preview_' if args.preview else 'masked_'}{input_path.name}'}")
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