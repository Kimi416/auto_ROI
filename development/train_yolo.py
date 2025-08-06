#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8を使用した皮膚病変検出モデルの学習スクリプト
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import argparse

def train_yolo_model(data_yaml_path, model_size='m', epochs=100, batch_size=16, img_size=640):
    """
    YOLOv8モデルを学習
    
    Args:
        data_yaml_path: データセット設定ファイルのパス
        model_size: モデルサイズ ('n', 's', 'm', 'l', 'x')
        epochs: エポック数
        batch_size: バッチサイズ
        img_size: 入力画像サイズ
    """
    
    # GPUが利用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    
    # モデルを初期化
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    
    print(f"\n=== YOLOv8{model_size.upper()} モデルの学習を開始 ===")
    print(f"データセット: {data_yaml_path}")
    print(f"エポック数: {epochs}")
    print(f"バッチサイズ: {batch_size}")
    print(f"画像サイズ: {img_size}")
    
    # 学習を実行
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=4,
        project='runs/detect',
        name='skin_lesion_detection',
        patience=50,  # 早期終了のための忍耐値
        save=True,
        save_period=10,  # 10エポックごとに保存
        plots=True,  # 学習曲線をプロット
        mosaic=1.0,  # モザイク拡張
        mixup=0.0,   # Mixup拡張
        copy_paste=0.0,  # Copy-paste拡張
        degrees=10.0,  # 回転拡張の角度
        translate=0.1,  # 平行移動拡張
        scale=0.5,   # スケール拡張
        shear=0.0,   # せん断拡張
        perspective=0.0,  # 透視変換拡張
        flipud=0.0,  # 上下反転確率
        fliplr=0.5,  # 左右反転確率
        hsv_h=0.015,  # HSV-Hue拡張
        hsv_s=0.7,    # HSV-Saturation拡張
        hsv_v=0.4,    # HSV-Value拡張
        amp=True,  # 自動混合精度トレーニング
        verbose=True
    )
    
    print("\n✅ 学習が完了しました！")
    print(f"モデルの保存先: runs/detect/skin_lesion_detection/weights/best.pt")
    
    return results

def validate_model(model_path, data_yaml_path):
    """学習済みモデルの検証"""
    model = YOLO(model_path)
    
    # 検証を実行
    results = model.val(
        data=data_yaml_path,
        split='val',
        save_json=True,
        save_hybrid=True,
        conf=0.25,
        iou=0.45,
        max_det=300,
        plots=True
    )
    
    print("\n=== 検証結果 ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8皮膚病変検出モデルの学習')
    parser.add_argument('--data', type=str, default='yolo_dataset/dataset.yaml',
                        help='データセット設定ファイルのパス')
    parser.add_argument('--model', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='モデルサイズ (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='学習エポック数')
    parser.add_argument('--batch', type=int, default=16,
                        help='バッチサイズ')
    parser.add_argument('--img-size', type=int, default=640,
                        help='入力画像サイズ')
    parser.add_argument('--validate', action='store_true',
                        help='学習後に検証を実行')
    
    args = parser.parse_args()
    
    # データセットパスを絶対パスに変換
    data_path = Path(args.data).resolve()
    
    if not data_path.exists():
        print(f"エラー: データセット設定ファイルが見つかりません: {data_path}")
        print("まず yolo_setup.py を実行してください")
        return
    
    # 学習を実行
    results = train_yolo_model(
        data_yaml_path=str(data_path),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    # 検証を実行（オプション）
    if args.validate:
        best_model_path = Path('runs/detect/skin_lesion_detection/weights/best.pt')
        if best_model_path.exists():
            validate_model(str(best_model_path), str(data_path))
        else:
            print("検証用のモデルが見つかりません")

if __name__ == "__main__":
    main()