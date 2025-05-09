import os

class Config:
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB限制
    
    MODEL_PATH = 'models/best_model_resnet50.pth'

    PORT = 5000
    