import os

class Config:
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 上传单文件限制
    MAX_UPLOAD_FOLDER_SIZE = 100 * 1024 * 1024  # 上传文件夹限制
    MODEL_IDLE_TIME = 0  # 模型空闲卸载时间（秒）
    
    MODEL_PATH = 'models/final_model_resnet50.pth'
    PORT = 8000
    