#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_preview.jpgの病変検出（シンプル版）
dlibなしでも実行可能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def detect_lesions_in_test_preview():
    """
    test_preview.jpgから病変を検出し、結果を可視化
    """
    print("🎯 test_preview.jpg 病変検出開始")
    print("=" * 60)
    
    # ファイルパス
    image_path = "test_preview.jpg"
    model_path = "fast_lesion_training/training_runs/fast_lesion_20250806_095404/weights/best.pt"
    
    # モデルロード
    model = YOLO(model_path)
    print(f"✅ モデルロード完了")
    
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 画像読み込みエラー: {image_path}")
        return
    
    print(f"✅ 画像読み込み完了: {image.shape[1]} x {image.shape[0]}")
    
    # BGR -> RGB変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # クラス定義
    pad_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    class_names_jp = {
        'ACK': '日光角化症',
        'BCC': '基底細胞癌', 
        'MEL': '悪性黒色腫',
        'NEV': '色素性母斑',
        'SCC': '有棘細胞癌',
        'SEK': '脂漏性角化症'
    }
    
    # 色定義（BGR）
    colors_bgr = {
        'ACK': (0, 255, 0),      # 緑
        'BCC': (0, 0, 255),      # 赤
        'MEL': (255, 0, 255),    # マゼンタ
        'NEV': (255, 255, 0),    # シアン
        'SCC': (0, 165, 255),    # オレンジ
        'SEK': (128, 0, 128)     # 紫
    }
    
    # 複数の信頼度で推論（超低信頼度も含む）
    confidence_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
    best_detections = []
    best_conf = 0.1
    
    for conf_threshold in confidence_levels:
        print(f"🔍 信頼度 {conf_threshold} で推論実行...")
        results = model(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = pad_classes[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detections.append({
                        'class': class_name,
                        'class_jp': class_names_jp[class_name],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        print(f"  検出数: {len(detections)}")
        if detections:
            best_detections = detections
            best_conf = conf_threshold
            break
    
    # 結果表示
    print(f"\\n📊 最終検出結果 (信頼度: {best_conf}):")
    if best_detections:
        print(f"  検出された病変数: {len(best_detections)}")
        
        # 信頼度でソート
        best_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 可視化用画像作成
        annotated_image = image.copy()
        
        for i, det in enumerate(best_detections, 1):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            class_name_jp = det['class_jp']
            conf = det['confidence']
            color = colors_bgr[class_name]
            
            print(f"  {i}. {class_name_jp} ({class_name})")
            print(f"     信頼度: {conf:.3f}")
            print(f"     位置: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"     サイズ: {x2-x1}px × {y2-y1}px")
            print()
            
            # バウンディングボックス描画（太線）
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 5)
            
            # ラベル描画
            label = f"{i}. {class_name} {conf:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # ラベル位置調整
            label_y = y1 - 15
            if label_y < 30:
                label_y = y2 + 40
                
            # ラベル背景
            cv2.rectangle(annotated_image, 
                        (x1, label_y - label_size[1] - 10), 
                        (x1 + label_size[0] + 10, label_y + 5), 
                        color, -1)
            
            # ラベルテキスト
            cv2.putText(annotated_image, label, (x1 + 5, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 中央に番号表示
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_image, (center_x, center_y), 25, color, -1)
            cv2.putText(annotated_image, str(i), (center_x-12, center_y+12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # 結果保存
        cv2.imwrite("test_preview_lesion_detection_result.jpg", annotated_image)
        print("💾 病変検出結果保存: test_preview_lesion_detection_result.jpg")
        
        # 比較可視化作成
        create_comparison_visualization(image_rgb, annotated_image, best_detections)
        
    else:
        print("  検出された病変はありませんでした")
    
    return best_detections

def create_comparison_visualization(original_rgb, annotated_bgr, detections):
    """
    比較可視化を作成
    """
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # 2列比較表示
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 元画像
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image (test_preview.jpg)', fontsize=16, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # 病変検出結果
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f'Lesion Detection Results\\n({len(detections)} lesions detected)', 
                     fontsize=16, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # 凡例作成
    if detections:
        colors_rgb = {
            'ACK': (0, 1, 0),        # 緑
            'BCC': (1, 0, 0),        # 赤
            'MEL': (1, 0, 1),        # マゼンタ
            'NEV': (0, 1, 1),        # シアン
            'SCC': (1, 0.5, 0),      # オレンジ
            'SEK': (0.5, 0, 0.5)     # 紫
        }
        
        legend_elements = []
        for i, det in enumerate(detections, 1):
            class_name = det['class']
            class_name_jp = det['class_jp']
            conf = det['confidence']
            color = colors_rgb[class_name]
            
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, 
                                               label=f"{i}. {class_name_jp} ({class_name}) - {conf:.3f}"))
        
        axes[1].legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1, 1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig("test_preview_comparison_visualization.png", dpi=300, bbox_inches='tight')
    print("💾 比較可視化保存: test_preview_comparison_visualization.png")
    plt.show()

if __name__ == "__main__":
    detections = detect_lesions_in_test_preview()
    
    print(f"\\n🎉 test_preview.jpg 病変検出完了!")
    if detections:
        print(f"📊 検出結果まとめ:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_jp']} - 信頼度: {det['confidence']:.3f}")
    else:
        print("❌ 病変は検出されませんでした")