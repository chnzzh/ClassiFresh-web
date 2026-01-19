#!/bin/bash

# Meat Freshness Classification Application Startup Script

echo "Starting Meat Freshness Classification Server..."

# 创建日志目录
mkdir -p /tmp/gunicorn

# 使用 gunicorn 启动生产服务器
# -w 4: 4个工作进程
# -b 0.0.0.0:8000: 绑定到所有网络接口的8000端口
# --timeout 120: 请求超时时间（推理可能需要较长时间）
# --access-logfile: 访问日志输出到文件
# --error-logfile: 错误日志输出到文件

gunicorn -w 4 \
    -b 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /tmp/gunicorn/access.log \
    --error-logfile /tmp/gunicorn/error.log \
    app:app
