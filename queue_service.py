import asyncio
import uuid
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import aioredis
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    image_data: bytes
    content_type: str
    status: TaskStatus
    created_at: float
    result: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class AsyncQueueService:
    """使用 asyncio.Queue 的隊列服務（適合單機）"""
    
    def __init__(self, max_queue_size: int = 100, max_workers: int = 2):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.max_workers = max_workers
        self.workers = []
        self.is_running = False
        
    async def start_workers(self, detector_func):
        """啟動工作者"""
        self.is_running = True
        self.detector_func = detector_func
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} workers")
    
    async def stop_workers(self):
        """停止工作者"""
        self.is_running = False
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("All workers stopped")
    
    async def submit_task(self, image_data: bytes, content_type: str) -> str:
        """提交任務到隊列"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            image_data=image_data,
            content_type=content_type,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )
        
        try:
            # 如果隊列滿了，立即拋出異常
            self.queue.put_nowait(task)
            self.tasks[task_id] = task
            logger.info(f"Task {task_id} submitted to queue")
            return task_id
        except asyncio.QueueFull:
            raise Exception("隊列已滿，請稍後再試")
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """獲取任務狀態"""
        return self.tasks.get(task_id)
    
    async def get_queue_info(self) -> Dict[str, Any]:
        """獲取隊列信息"""
        pending_count = sum(1 for task in self.tasks.values() 
                          if task.status == TaskStatus.PENDING)
        processing_count = sum(1 for task in self.tasks.values() 
                             if task.status == TaskStatus.PROCESSING)
        
        return {
            "queue_size": self.queue.qsize(),
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "total_tasks": len(self.tasks),
            "workers": len(self.workers),
            "is_running": self.is_running
        }
    
    async def _worker(self, worker_name: str):
        """工作者協程"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # 從隊列取出任務
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # 更新任務狀態
                task.status = TaskStatus.PROCESSING
                start_time = time.time()
                
                logger.info(f"Worker {worker_name} processing task {task.id}")
                
                try:
                    # 執行檢測
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.detector_func, task.image_data
                    )
                    
                    # 任務完成
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.processing_time = time.time() - start_time
                    
                    logger.info(f"Task {task.id} completed in {task.processing_time:.3f}s")
                    
                except Exception as e:
                    # 任務失敗
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    task.processing_time = time.time() - start_time
                    
                    logger.error(f"Task {task.id} failed: {str(e)}")
                
                finally:
                    # 標記任務完成
                    self.queue.task_done()
                    
            except asyncio.TimeoutError:
                # 沒有任務，繼續等待
                continue
            except asyncio.CancelledError:
                # 工作者被取消
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {str(e)}")
        
        logger.info(f"Worker {worker_name} stopped")

class RedisQueueService:
    """使用 Redis 的隊列服務（適合分佈式）"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 queue_name: str = "yolo_detection", 
                 max_workers: int = 2):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.task_prefix = f"{queue_name}:task:"
        self.max_workers = max_workers
        self.redis = None
        self.workers = []
        self.is_running = False
    
    async def connect(self):
        """連接 Redis"""
        self.redis = await aioredis.from_url(self.redis_url)
        logger.info(f"Connected to Redis: {self.redis_url}")
    
    async def disconnect(self):
        """斷開 Redis 連接"""
        if self.redis:
            await self.redis.close()
    
    async def start_workers(self, detector_func):
        """啟動工作者"""
        if not self.redis:
            await self.connect()
        
        self.is_running = True
        self.detector_func = detector_func
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} Redis workers")
    
    async def stop_workers(self):
        """停止工作者"""
        self.is_running = False
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("All Redis workers stopped")
    
    async def submit_task(self, image_data: bytes, content_type: str) -> str:
        """提交任務到 Redis 隊列"""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "content_type": content_type,
            "status": TaskStatus.PENDING.value,
            "created_at": time.time()
        }
        
        # 將圖片數據和任務信息分別存儲
        task_key = f"{self.task_prefix}{task_id}"
        image_key = f"{task_key}:image"
        
        # 使用 pipeline 提高性能
        pipe = self.redis.pipeline()
        pipe.hset(task_key, mapping=task_data)
        pipe.set(image_key, image_data)
        pipe.lpush(self.queue_name, task_id)
        pipe.expire(task_key, 3600)  # 1小時過期
        pipe.expire(image_key, 3600)
        await pipe.execute()
        
        logger.info(f"Task {task_id} submitted to Redis queue")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取任務狀態"""
        task_key = f"{self.task_prefix}{task_id}"
        task_data = await self.redis.hgetall(task_key)
        
        if not task_data:
            return None
        
        # 解碼 Redis 返回的字節數據
        result = {}
        for key, value in task_data.items():
            key = key.decode() if isinstance(key, bytes) else key
            value = value.decode() if isinstance(value, bytes) else value
            
            if key in ['created_at', 'processing_time']:
                result[key] = float(value) if value else None
            elif key == 'result':
                result[key] = json.loads(value) if value else None
            else:
                result[key] = value
        
        return result
    
    async def get_queue_info(self) -> Dict[str, Any]:
        """獲取隊列信息"""
        queue_size = await self.redis.llen(self.queue_name)
        
        # 計算不同狀態的任務數量
        pattern = f"{self.task_prefix}*"
        task_keys = []
        async for key in self.redis.scan_iter(match=pattern):
            if not key.decode().endswith(':image'):
                task_keys.append(key)
        
        status_counts = {status.value: 0 for status in TaskStatus}
        for key in task_keys:
            status = await self.redis.hget(key, 'status')
            if status:
                status = status.decode()
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "queue_size": queue_size,
            "pending_tasks": status_counts[TaskStatus.PENDING.value],
            "processing_tasks": status_counts[TaskStatus.PROCESSING.value],
            "completed_tasks": status_counts[TaskStatus.COMPLETED.value],
            "failed_tasks": status_counts[TaskStatus.FAILED.value],
            "total_tasks": len(task_keys),
            "workers": len(self.workers),
            "is_running": self.is_running
        }
    
    async def _worker(self, worker_name: str):
        """Redis 工作者協程"""
        logger.info(f"Redis worker {worker_name} started")
        
        while self.is_running:
            try:
                # 從 Redis 隊列取出任務
                task_id = await self.redis.brpop(self.queue_name, timeout=1)
                
                if not task_id:
                    continue
                
                task_id = task_id[1].decode()  # Redis 返回 (queue_name, task_id)
                task_key = f"{self.task_prefix}{task_id}"
                image_key = f"{task_key}:image"
                
                # 更新任務狀態
                await self.redis.hset(task_key, "status", TaskStatus.PROCESSING.value)
                start_time = time.time()
                
                logger.info(f"Redis worker {worker_name} processing task {task_id}")
                
                try:
                    # 獲取圖片數據
                    image_data = await self.redis.get(image_key)
                    if not image_data:
                        raise Exception("圖片數據不存在")
                    
                    # 執行檢測
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.detector_func, image_data
                    )
                    
                    # 任務完成
                    processing_time = time.time() - start_time
                    await self.redis.hset(task_key, mapping={
                        "status": TaskStatus.COMPLETED.value,
                        "result": json.dumps(result),
                        "processing_time": processing_time
                    })
                    
                    logger.info(f"Redis task {task_id} completed in {processing_time:.3f}s")
                    
                except Exception as e:
                    # 任務失敗
                    processing_time = time.time() - start_time
                    await self.redis.hset(task_key, mapping={
                        "status": TaskStatus.FAILED.value,
                        "error": str(e),
                        "processing_time": processing_time
                    })
                    
                    logger.error(f"Redis task {task_id} failed: {str(e)}")
                
                finally:
                    # 清理圖片數據（可選）
                    await self.redis.delete(image_key)
                    
            except asyncio.CancelledError:
                # 工作者被取消
                break
            except Exception as e:
                logger.error(f"Redis worker {worker_name} error: {str(e)}")
        
        logger.info(f"Redis worker {worker_name} stopped")
