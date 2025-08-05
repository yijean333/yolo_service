# YOLOv8 物件偵測服務

透過 FastAPI + Ultralytics YOLOv8 建立的圖片物件偵測 API，支援非同步任務佇列。

---

## 功能

* 上傳圖片，返回任務 ID
* 透過任務 ID 查詢辨識結果
* 支援 CPU/GPU 推論
* Docker 容器化部署
* 可搭配 ngrok 讓手機 APP 呼叫外網 API

---

## 快速使用

1. **Clone 專案**

```bash
git clone https://github.com/yijean333/yolo_service.git
cd yolo_service
```

2. **用 Docker 建置並執行**

```bash
docker build -t yolo_service .
docker run -p 8000:8000 yolo_service
```

3. **用 ngrok 暴露服務**

```bash
ngrok config add-authtoken <你的-authtoken-字串> # 推薦但不必要
ngrok http http://localhost:8000
```

4. **API 簡介**

* POST `/predict`：上傳圖片，回傳任務 ID
* GET `/task/{task_id}`：查詢辨識結果 (JSON格式)
* **GET `/task/{task_id}/image`：獲取帶檢測框的圖片 (JPEG格式)**
* 更多 API 功能和詳細說明，請參考: `你的-ngrok網址/docs`
  
---

## 注意事項

* 請自行準備 `best.pt` 模型權重放到專案根目錄
* Docker 裡安裝的 PyTorch 與 torchvision 版本要配合
* 請替換 ngrok 網址為你自己產生的外網 URL

