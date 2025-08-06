#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
既存モデルへのBCCデータ追加学習
Phase 2完了モデル（mAP50: 0.621）に体幹部BCCデータを追加
"""

from ultralytics import YOLO
import torch
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

class IncrementalBCCTrainer:
    def __init__(self):
        self.base_model = 'runs/detect/optimal_stable_phase2_final/weights/best.pt'
        self.trunk_bcc_dir = Path('trunk_bcc_images')  # 体幹部BCC画像フォルダ
        self.augmented_dataset = Path('augmented_dataset')
        
    def prepare_incremental_dataset(self):
        """既存データセットに新しいBCCデータを追加"""
        print("📂 データセット準備中...")
        
        # 既存のデータセットをコピー
        if self.augmented_dataset.exists():
            shutil.rmtree(self.augmented_dataset)
        
        shutil.copytree('yolo_dataset', self.augmented_dataset)
        
        # 体幹部BCC画像の追加準備
        bcc_train_dir = self.augmented_dataset / 'train' / 'images' / 'bcc_trunk'
        bcc_train_dir.mkdir(parents=True, exist_ok=True)
        
        bcc_label_dir = self.augmented_dataset / 'train' / 'labels' / 'bcc_trunk'
        bcc_label_dir.mkdir(parents=True, exist_ok=True)
        
        return bcc_train_dir, bcc_label_dir
        
    def extract_bcc_regions(self, image_path, output_dir, label_dir):
        """体幹部BCC領域の抽出（手動アノテーション必要）"""
        print(f"🔍 処理中: {image_path}")
        
        # ここでは仮のBCC検出
        # 実際には手動でアノテーションするか、別の検出器を使用
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
            
        height, width = img.shape[:2]
        
        # 仮のBCC領域（実際には適切なアノテーションが必要）
        # YOLOフォーマット: class_id center_x center_y width height (正規化座標)
        bcc_class_id = 5  # Basal cell carcinomaのクラスID
        
        # 画像を保存
        output_path = output_dir / f"bcc_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), img)
        
        # ラベルファイル作成（手動アノテーション必要）
        label_path = label_dir / f"bcc_{image_path.stem}.txt"
        with open(label_path, 'w') as f:
            # 仮のアノテーション（実際のBCC位置に置き換える必要）
            f.write(f"{bcc_class_id} 0.5 0.5 0.2 0.2\n")
            
        return 1
        
    def create_augmented_yaml(self):
        """拡張データセット用のYAMLファイル作成"""
        yaml_content = f"""path: {self.augmented_dataset.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 8
names: ['ADM', 'Ephelis', 'Melasma', 'Solar lentigo', 'Nevus', 'Basal cell carcinoma', 'Seborrheic keratosis', 'Malignant melanoma']
"""
        
        with open('augmented_lesion_detection.yaml', 'w') as f:
            f.write(yaml_content)
            
    def incremental_train(self):
        """既存モデルへの追加学習"""
        print("🚀 既存モデルへの追加学習開始")
        print(f"ベースモデル: {self.base_model}")
        
        # 既存の学習済みモデルを読み込み
        model = YOLO(self.base_model)
        
        # デバイス設定
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"デバイス: {device}")
        
        print("\n⚙️ 追加学習設定:")
        print("- 低学習率で既存知識を保持")
        print("- BCCサンプルを重点的に学習")
        print("- 既存の検出性能を維持")
        
        results = model.train(
            data='augmented_lesion_detection.yaml',
            epochs=10,  # 少ないエポックで微調整
            batch=6,
            device=device,
            lr0=0.00001,  # 非常に低い学習率で既存の知識を保持
            lrf=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1,
            # データ拡張は控えめに
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            translate=0.05,
            scale=0.3,
            mosaic=0.5,  # モザイクは控えめ
            mixup=0.0,   # ミックスアップ無効
            # BCCクラスに焦点
            cls=3.0,     # クラス損失を重視
            # 保存設定
            save=True,
            save_period=2,
            val=True,
            plots=True,
            exist_ok=True,
            project='runs/detect',
            name='incremental_bcc',
            verbose=True,
            patience=5,
            # 既存の重みを最大限活用
            pretrained=False,  # 事前学習済みモデルなので不要
            resume=False       # 新規学習として扱う
        )
        
        return results
        
    def analyze_improvement(self):
        """改善度分析"""
        print("\n📊 BCCクラスの改善度分析")
        
        # 元のモデルでBCCテスト
        original_model = YOLO(self.base_model)
        original_results = original_model.val(
            data='lesion_detection.yaml',
            split='test',
            verbose=False
        )
        
        # 新しいモデルでBCCテスト
        new_model = YOLO('runs/detect/incremental_bcc/weights/best.pt')
        new_results = new_model.val(
            data='augmented_lesion_detection.yaml',
            split='test',
            verbose=False
        )
        
        # BCC（クラス5）の性能比較
        print("Basal cell carcinoma検出性能:")
        print(f"  元のモデル: mAP50={original_results.box.maps[5]:.3f}")
        print(f"  追加学習後: mAP50={new_results.box.maps[5]:.3f}")
        
def main():
    print("🔬 体幹部BCCデータによる既存モデル改善")
    print("="*50)
    
    trainer = IncrementalBCCTrainer()
    
    # 1. データセット準備
    bcc_train_dir, bcc_label_dir = trainer.prepare_incremental_dataset()
    
    # 2. 体幹部BCC画像の処理
    if trainer.trunk_bcc_dir.exists():
        bcc_count = 0
        for img_path in trainer.trunk_bcc_dir.glob("*.jpg"):
            bcc_count += trainer.extract_bcc_regions(img_path, bcc_train_dir, bcc_label_dir)
        print(f"✅ {bcc_count}個のBCC画像を追加")
    else:
        print(f"⚠️ {trainer.trunk_bcc_dir} フォルダに体幹部BCC画像を配置してください")
        print("画像配置後、各画像のBCC位置をアノテーションする必要があります")
        return
    
    # 3. 拡張データセットYAML作成
    trainer.create_augmented_yaml()
    
    # 4. 追加学習実行
    print("\n既存モデルに追加学習を実行しますか？")
    print("注意: 体幹部BCC画像には適切なアノテーション（バウンディングボックス）が必要です")
    print("続行する場合は、このスクリプトを再実行してください")
    
    # 実際の学習実行（アノテーション済みの場合）
    # results = trainer.incremental_train()
    # if results:
    #     trainer.analyze_improvement()

if __name__ == "__main__":
    main()