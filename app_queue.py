from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
import asyncio
import uvicorn
from queue_service import AsyncQueueService, TaskStatus

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時初始化
    global model, device, queue_service
    
    print("正在載入模型...")
    # 自動選擇可用的模型檔案
    import os
    if os.path.exists("best.pt") and os.path.getsize("best.pt") > 1000:  # 檢查檔案大小
        model_path = "best.pt"
        print("使用自訓練模型: best.pt")
    elif os.path.exists("yolov8n.pt"):
        model_path = "yolov8n.pt" 
        print("使用預訓練模型: yolov8n.pt")
    else:
        raise FileNotFoundError("找不到有效的模型檔案")
    
    model = YOLO(model_path)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已載入到設備: {device}")
    
    if device.type == 'cpu':
        torch.set_num_threads(4)
    
    # 初始化隊列服務（2個工作者，最大隊列100）
    queue_service = AsyncQueueService(max_queue_size=100, max_workers=2)
    await queue_service.start_workers(run_detection_sync)
    
    print("隊列服務已啟動")
    
    yield
    
    # 關閉時清理
    if queue_service:
        await queue_service.stop_workers()
    print("隊列服務已停止")

app = FastAPI(title="YOLOv8 隊列檢測服務", version="1.0.0", lifespan=lifespan)

# 開放跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# 全局變數
model = None
device = None
queue_service = None

def resize_if_needed(img, max_dim=640):
    """優化的圖片縮放函數"""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img

def run_detection_sync(image_data: bytes):
    """同步檢測函數（供隊列工作者使用）"""
    try:
        # 解碼圖片
        arr = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise Exception("無法解碼圖片")

        start = time.time()
        
        # 圖片預處理
        img_small = resize_if_needed(img, max_dim=640)
        after_resize = time.time()
        
        # 模型推理
        with torch.inference_mode():
            results = model.predict(
                source=img_small, 
                save=False, 
                verbose=False,
                stream=False,
                device=device,
                half=False,
                augment=False,
                agnostic_nms=True,
                max_det=300,
                conf=0.25,
                iou=0.45,
            )[0]
        
        after_infer = time.time()

        # 處理結果
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []
            classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else []
            
            for i, box in enumerate(boxes):
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(classes[i]) if i < len(classes) else -1
                label = model.names.get(cls_id, str(cls_id))
                
                detections.append({
                    "box": box.tolist(),
                    "confidence": conf,
                    "class_id": cls_id,
                    "label": label,
                })
        
        end = time.time()
        print(f"[detection] resize: {after_resize - start:.3f}s, infer: {after_infer - after_resize:.3f}s, total: {end - start:.3f}s, detections: {len(detections)}")
        
        return {"success": True, "detections": detections}
        
    except Exception as e:
        print(f"檢測錯誤: {str(e)}")
        raise e



@app.get("/")
def health():
    return PlainTextResponse("YOLOv8 隊列檢測服務運行中")

@app.get("/status")
async def status():
    """服務狀態檢查"""
    queue_info = await queue_service.get_queue_info() if queue_service else {}
    
    return {
        "status": "ready",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "queue_info": queue_info
    }

@app.post("/predict")
async def predict_async(file: UploadFile = File(...)):
    """異步預測端點 - 立即返回任務ID"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="需要圖片檔")
    
    try:
        # 讀取圖片數據
        contents = await file.read()
        
        # 提交任務到隊列
        task_id = await queue_service.submit_task(contents, file.content_type)
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "任務已提交到隊列，請使用 task_id 查詢結果"
        }
        
    except Exception as e:
        print(f"提交任務錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    """獲取任務結果"""
    task = await queue_service.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    response = {
        "task_id": task_id,
        "status": task.status.value,
        "created_at": task.created_at,
        "processing_time": task.processing_time
    }
    
    if task.status == TaskStatus.COMPLETED:
        response["result"] = task.result
    elif task.status == TaskStatus.FAILED:
        response["error"] = task.error
    
    return response

@app.post("/predict-sync")
async def predict_sync(file: UploadFile = File(...)):
    """同步預測端點 - 等待結果完成"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="需要圖片檔")
    
    try:
        # 讀取圖片數據
        contents = await file.read()
        
        # 提交任務到隊列
        task_id = await queue_service.submit_task(contents, file.content_type)
        
        # 輪詢等待結果（最多等待30秒）
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            task = await queue_service.get_task_status(task_id)
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise HTTPException(status_code=500, detail=task.error)
            
            # 等待0.1秒後再檢查
            await asyncio.sleep(0.1)
        
        # 超時
        raise HTTPException(status_code=408, detail="處理超時，請使用異步API")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"同步預測錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue/info")
async def get_queue_info():
    """獲取隊列狀態信息"""
    return await queue_service.get_queue_info()

if __name__ == "__main__":
    print("啟動 YOLOv8 隊列檢測服務...")
    uvicorn.run(app, host="0.0.0.0", port=8000)