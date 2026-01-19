# coding=utf-8

import os
import time
import platform
import threading
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
logger.info(f"Max upload folder size: {app.config['MAX_UPLOAD_FOLDER_SIZE'] / (1024*1024):.1f} MB")
logger.info(f"Model idle time: {app.config['MODEL_IDLE_TIME']} seconds")

# 模型管理变量
model = None
model_lock = threading.Lock()
unload_timer = None


# 加载模型
def load_model():
    logger.info("Starting to load model...")
    # 构建模型结构
    model = resnet50(weights=None)
    num_classes = 3  # 根据你的训练代码设置
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 加载训练好的权重
    model_path = app.config['MODEL_PATH']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model from {model_path} on device {device}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
    return model


# 卸载模型释放内存
def unload_model():
    global model, unload_timer
    with model_lock:
        if model is not None:
            logger.info("Unloading model to free memory...")
            del model
            model = None
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded successfully")
        if unload_timer is not None:
            unload_timer.cancel()
            unload_timer = None


# 获取或加载模型
def get_model():
    global model, unload_timer
    with model_lock:
        # 取消之前的卸载定时器
        if unload_timer is not None:
            unload_timer.cancel()
            unload_timer = None
        
        # 如果模型未加载，则加载
        if model is None:
            logger.info("Model not loaded, loading now...")
            model = load_model()
        else:
            logger.debug("Using already loaded model")
        
        return model


# 设置延迟卸载定时器
def schedule_unload():
    global unload_timer
    idle_time = app.config['MODEL_IDLE_TIME']
    
    # 如果 idle_time 为 0，则不启用自动卸载功能
    if idle_time == 0:
        logger.info("Model auto-unload disabled (MODEL_IDLE_TIME=0)")
        return
    
    with model_lock:
        # 取消之前的定时器
        if unload_timer is not None:
            unload_timer.cancel()
        
        # 设置新的定时器
        unload_timer = threading.Timer(idle_time, unload_model)
        unload_timer.daemon = True
        unload_timer.start()
        logger.info(f"Model will be unloaded after {idle_time} seconds of idle time")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 类别标签 (中英双语)
class_labels = {
    "zh": ["新鲜", "次新鲜", "腐败"],
    "en": ["Fresh", "Semi-fresh", "Rotten"]
}

# 获取设备信息
def get_device_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"GPU: {device_name}"
    else:
        # 获取CPU型号
        if platform.system() == 'Darwin':  # macOS
            import subprocess
            try:
                cpu_model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                return f"CPU: {cpu_model}"
            except:
                return f"CPU: {platform.processor()}"
        else:
            return f"CPU: {platform.processor()}"

device_info = get_device_info()
logger.info(f"Device info: {device_info}")


# 清理上传文件夹，删除最旧的文件直到总大小低于限制
def cleanup_upload_folder():
    upload_folder = app.config['UPLOAD_FOLDER']
    max_size = app.config['MAX_UPLOAD_FOLDER_SIZE']
    
    # 获取所有文件及其信息
    files = []
    total_size = 0
    
    for filename in os.listdir(upload_folder):
        filepath = os.path.join(upload_folder, filename)
        if os.path.isfile(filepath):
            file_stat = os.stat(filepath)
            files.append({
                'path': filepath,
                'size': file_stat.st_size,
                'mtime': file_stat.st_mtime
            })
            total_size += file_stat.st_size
    
    # 如果总大小超过限制，删除最旧的文件
    if total_size > max_size:
        logger.info(f"Upload folder size ({total_size / (1024*1024):.2f} MB) exceeds limit ({max_size / (1024*1024):.1f} MB), cleaning up...")
        # 按修改时间排序，最旧的在前面
        files.sort(key=lambda x: x['mtime'])
        
        deleted_count = 0
        for file_info in files:
            if total_size <= max_size:
                break
            
            try:
                os.remove(file_info['path'])
                total_size -= file_info['size']
                deleted_count += 1
                logger.info(f"Deleted: {os.path.basename(file_info['path'])} ({file_info['size'] / 1024:.1f} KB)")
            except Exception as e:
                logger.error(f"Error deleting {file_info['path']}: {e}")
        
        logger.info(f"Cleanup complete: deleted {deleted_count} files, current size: {total_size / (1024*1024):.2f} MB")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # 如果用户没有选择文件
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File uploaded: {filename} ({os.path.getsize(filepath) / 1024:.1f} KB)")
            
            # 清理上传文件夹，防止无限增长
            cleanup_upload_folder()

            # 获取模型（如果未加载则自动加载）
            current_model = get_model()

            # 图像分类
            img = Image.open(filepath).convert('RGB')
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            # 记录推理时间
            start_time = time.time()
            with torch.no_grad():
                out = current_model(batch_t)
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            logger.info(f"Inference completed in {inference_time:.2f} ms")

            # 获取预测结果
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 3)

            results = []
            for i in range(top5_prob.size(0)):
                results.append({
                    'label_zh': class_labels["zh"][top5_catid[i]],
                    'label_en': class_labels["en"][top5_catid[i]],
                    'probability': f"{top5_prob[i].item() * 100:.2f}%"
                })
            
            logger.info(f"Prediction result: {class_labels['en'][top5_catid[0]]} ({top5_prob[0].item() * 100:.2f}%)")
            
            # 设置延迟卸载定时器
            schedule_unload()

            return render_template('index.html',
                                   filename=filename,
                                   results=results,
                                   inference_time=f"{inference_time:.2f}",
                                   device_info=device_info)

    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting Meat Freshness Classification Application")
    logger.info("="*60)
    logger.info(f"Server starting on http://0.0.0.0:{app.config['PORT']}")
    logger.info("For production, use: gunicorn -w 4 -b 0.0.0.0:8000 app:app")
    
    # 开发模式使用 Flask 内置服务器
    app.run(host='0.0.0.0', port=app.config['PORT'])