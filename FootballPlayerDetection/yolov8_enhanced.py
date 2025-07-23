import cv2
import torch
import numpy as np
from ultralytics import YOLO
# Removed unused or unresolved import
import matplotlib.pyplot as plt
from IPython.display import display, Image

def plot_results(results):
    """可视化训练结果"""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(results['metrics/precision'], label='Precision')
    plt.plot(results['metrics/recall'], label='Recall')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(results['metrics/mAP_0.5'], label='mAP@0.5')
    plt.plot(results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
    plt.title('mAP Metrics')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(results['train/box_loss'], label='Train Box Loss')
    plt.plot(results['val/box_loss'], label='Val Box Loss')
    plt.title('Box Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(results['train/cls_loss'], label='Train Cls Loss')
    plt.plot(results['val/cls_loss'], label='Val Cls Loss')
    plt.title('Classification Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def show_detections(model, img_path):
    """显示检测结果"""
    results = model(img_path)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.axis('off')
        plt.show()

def train_model():
    # 初始化模型
    model = YOLO("yolov8s.pt")
    
    # 自定义训练配置
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        batch=16,
        imgsz=1024,
        name='yolov8s-football-enhanced',
        project='yolo-football-detection',
        
        # 图像增强配置
        augment=True,
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,    # 饱和度增强
        hsv_v=0.4,    # 亮度增强
        degrees=10.0, # 旋转角度
        translate=0.1, # 平移
        scale=0.5,    # 缩放
        shear=2.0,    # 剪切
        perspective=0.001, # 透视变换
        flipud=0.5,   # 上下翻转概率
        fliplr=0.5,   # 左右翻转概率
        mosaic=1.0,   # 马赛克增强概率
        mixup=0.1,    # MixUp增强概率
        
        # 小目标优化
        box=7.5,      # box loss增益
        cls=0.5,      # cls loss增益
        dfl=1.5,      # dfl loss增益
        close_mosaic=10, # 最后10epoch关闭马赛克增强
        
        # 训练监控
        save_period=5, # 每5epoch保存一次
        device='0',   # 使用GPU 0
        workers=8,    # 数据加载线程
        single_cls=False,
        verbose=True
    )
    
    # 保存最佳模型
    best_model = YOLO(results.save_dir + '/weights/best.pt')
    
    # 可视化训练结果
    plot_results(results)
    
    # 示例检测
    test_img = 'dataset/test/images/0001.jpg'  # 修改为实际测试图片路径
    show_detections(best_model, test_img)
    
    return best_model

if __name__ == '__main__':
    trained_model = train_model()
    # 保存模型为TorchScript格式
    trained_model.export(format='torchscript')
