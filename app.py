import os
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

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# 加载模型
def load_model():
    # 构建模型结构
    model = resnet50(weights=None)
    num_classes = 3  # 根据你的训练代码设置
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 加载训练好的权重
    model_path = app.config['MODEL_PATH']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} on device {device}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


model = load_model()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 类别标签 (根据你的实际类别修改)
class_labels = ["新鲜", "次新鲜", "腐败"]


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

            # 图像分类
            img = Image.open(filepath).convert('RGB')
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                out = model(batch_t)

            # 获取预测结果
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 3)

            results = []
            for i in range(top5_prob.size(0)):
                results.append({
                    'label': class_labels[top5_catid[i]],
                    'probability': f"{top5_prob[i].item() * 100:.2f}%"
                })

            return render_template('index.html',
                                   filename=filename,
                                   results=results)

    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)