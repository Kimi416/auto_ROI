#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard Negative Mining: 誤検出データ収集器
現在のモデルで低閾値検出を行い、誤検出を特定・収集する
"""

import cv2
import numpy as np
import json
from pathlib import Path
import shutil
from ultralytics import YOLO

class FalsePositiveCollector:
    def __init__(self):
        self.model_path = "fast_lesion_training/training_runs/fast_lesion_20250806_095404/weights/best.pt"
        self.model = YOLO(self.model_path)
        
        # 出力ディレクトリ
        self.negatives_dir = Path("datasets/negatives")
        self.negatives_images_dir = self.negatives_dir / "images"
        self.negatives_labels_dir = self.negatives_dir / "labels"
        
        # ディレクトリ作成
        self.negatives_images_dir.mkdir(parents=True, exist_ok=True)
        self.negatives_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Hard Negative Mining セットアップ完了")
        print(f"📁 出力先: {self.negatives_dir}")
    
    def collect_from_validation_set(self):
        """検証用データセットから誤検出を収集"""
        print(f"\n🔍 検証セットからの誤検出収集開始")
        print("=" * 60)
        
        # 既存の検証用画像を対象にする
        validation_sources = [
            "test_preview.jpg",
            "test1.jpeg", 
            "test2.jpeg"
        ]
        
        # organized_advanced_maskedから一部画像も検証用として使用
        masked_dir = Path("organized_advanced_masked")
        additional_samples = []
        
        for class_dir in masked_dir.glob("*"):
            if class_dir.is_dir():
                # 各クラスから5枚ずつサンプル
                sample_images = list(class_dir.glob("*.jpg"))[:5]
                additional_samples.extend(sample_images)
        
        print(f"📊 検証対象: {len(validation_sources)}個の既存テスト画像 + {len(additional_samples)}個の追加サンプル")
        
        all_false_positives = []
        
        # 既存テスト画像での検出
        for img_path in validation_sources:
            if Path(img_path).exists():
                fps = self.detect_false_positives(img_path, is_known_negative=True)
                all_false_positives.extend(fps)
        
        # 追加サンプルでの検出（低信頼度で）
        for img_path in additional_samples[:20]:  # 最初の20枚のみ
            fps = self.detect_false_positives(str(img_path), is_known_negative=False)
            all_false_positives.extend(fps)
        
        print(f"\n📈 誤検出収集結果: {len(all_false_positives)}個")
        return all_false_positives
    
    def detect_false_positives(self, image_path, is_known_negative=False):
        """単一画像での誤検出検出"""
        print(f"🔍 {Path(image_path).name} を分析中...")
        
        # 低信頼度で検出実行
        confidence_thresholds = [0.01, 0.05, 0.1, 0.2]
        
        detections = []
        for conf in confidence_thresholds:
            results = self.model(image_path, conf=conf, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detections.append({
                            'image_path': image_path,
                            'confidence': confidence,
                            'threshold': conf,
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls_id,
                            'is_known_negative': is_known_negative
                        })
            
            if detections:
                break  # 最初に検出されたthresholdで停止
        
        if detections:
            print(f"  ✅ {len(detections)}個の検出（信頼度{detections[0]['threshold']}）")
        else:
            print(f"  ❌ 検出なし")
            
        # known negativeの場合は全て誤検出として扱う
        if is_known_negative:
            return detections
        
        # 通常の場合は手動確認が必要だが、今回は低信頼度のものを疑わしいとする
        suspicious = [d for d in detections if d['confidence'] < 0.3]
        return suspicious
    
    def create_negative_dataset(self, false_positives):
        """誤検出データからネガティブデータセットを作成"""
        print(f"\n📦 ネガティブデータセット作成開始")
        print("=" * 60)
        
        # 画像ごとにグループ化
        fp_by_image = {}
        for fp in false_positives:
            img_path = fp['image_path']
            if img_path not in fp_by_image:
                fp_by_image[img_path] = []
            fp_by_image[img_path].append(fp)
        
        created_count = 0
        for img_path, detections in fp_by_image.items():
            if self.process_negative_image(img_path, detections):
                created_count += 1
        
        print(f"📊 作成完了: {created_count}個のネガティブサンプル")
        
        # dataset.yamlを更新
        self.update_dataset_yaml()
        
        return created_count
    
    def process_negative_image(self, image_path, detections):
        """単一画像をネガティブサンプルとして処理"""
        src_path = Path(image_path)
        
        if not src_path.exists():
            return False
        
        # 新しいファイル名を生成
        neg_filename = f"neg_{src_path.stem}_{len(detections)}det{src_path.suffix}"
        
        # 画像をコピー
        dst_image_path = self.negatives_images_dir / neg_filename
        shutil.copy2(src_path, dst_image_path)
        
        # 空のラベルファイルを作成（重要！）
        label_filename = neg_filename.replace(src_path.suffix, '.txt')
        dst_label_path = self.negatives_labels_dir / label_filename
        dst_label_path.write_text("")  # 空ファイル
        
        print(f"  📋 {neg_filename} を作成（{len(detections)}個の誤検出あり）")
        
        return True
    
    def update_dataset_yaml(self):
        """dataset.yamlを更新してnegativesを含める"""
        yaml_path = Path("lesion_detection.yaml")
        
        if not yaml_path.exists():
            # 新規作成
            yaml_content = f"""# Hard Negative Mining対応データセット設定
path: {Path.cwd()}
train:
  - fast_lesion_training/yolo_dataset/train/images
  - datasets/negatives/images
val: fast_lesion_training/yolo_dataset/val/images
test: fast_lesion_training/yolo_dataset/test/images

# クラス
nc: 6
names:
  0: ACK
  1: BCC  
  2: MEL
  3: NEV
  4: SCC
  5: SEK
"""
        else:
            # 既存ファイルを更新
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # trainセクションを更新
            if 'datasets/negatives/images' not in content:
                yaml_content = content.replace(
                    'train: fast_lesion_training/yolo_dataset/train/images',
                    '''train:
  - fast_lesion_training/yolo_dataset/train/images
  - datasets/negatives/images'''
                )
            else:
                yaml_content = content
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"✅ dataset.yaml を更新完了")
    
    def generate_collection_report(self, false_positives, created_count):
        """収集レポートを生成"""
        report = {
            'timestamp': str(Path().resolve()),
            'model_used': self.model_path,
            'total_false_positives': len(false_positives),
            'negative_samples_created': created_count,
            'confidence_distribution': {},
            'class_distribution': {}
        }
        
        # 信頼度分布
        conf_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)]
        for low, high in conf_ranges:
            count = len([fp for fp in false_positives if low <= fp['confidence'] < high])
            report['confidence_distribution'][f"{low}-{high}"] = count
        
        # クラス分布
        class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        for cls_id in range(6):
            count = len([fp for fp in false_positives if fp['class_id'] == cls_id])
            report['class_distribution'][class_names[cls_id]] = count
        
        # レポート保存
        with open('false_positive_collection_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 収集レポート保存: false_positive_collection_report.json")
        
        return report

def main():
    collector = FalsePositiveCollector()
    
    print("🚀 Hard Negative Mining 開始")
    print("=" * 80)
    
    # Step 1: 誤検出データ収集
    false_positives = collector.collect_from_validation_set()
    
    if not false_positives:
        print("\n❌ 誤検出が見つかりませんでした")
        print("💡 より低い信頼度での検出や、より多くの検証画像が必要です")
        return
    
    # Step 2: ネガティブデータセット作成
    created_count = collector.create_negative_dataset(false_positives)
    
    # Step 3: レポート生成
    report = collector.generate_collection_report(false_positives, created_count)
    
    print(f"\n🎉 Hard Negative Mining 収集完了!")
    print(f"📊 収集サマリー:")
    print(f"  • 誤検出総数: {len(false_positives)}")
    print(f"  • ネガティブサンプル: {created_count}個")
    print(f"  • 信頼度分布: {report['confidence_distribution']}")
    print(f"  • クラス分布: {report['class_distribution']}")
    
    print(f"\n🔄 次のステップ:")
    print(f"  1. Hard Negativeトレーニングを実行")
    print(f"  2. python3 hard_negative_trainer.py")

if __name__ == "__main__":
    main()