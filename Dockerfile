FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔案
COPY requirements.txt .

# requirements.txt（不含 torch/torchvision）
RUN pip install --no-cache-dir \
  torch==2.1.0+cpu \
  torchvision==0.16.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
  
# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式檔案
COPY . .

# 暴露端口
EXPOSE 8000

# 預設使用 asyncio 版本，可以透過環境變數切換
ENV APP_TYPE=queue
ENV WORKERS=2
ENV MAX_QUEUE_SIZE=100

# 啟動命令
CMD ["python", "app_queue.py"]
