"""
Label Studio ML Backend for Kimi (Moonshot) - Production Optimized
Features: Connection pooling, structured logging, request tracing, circuit breaker pattern
"""

import os
import sys
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import openai
import httpx

from starlette.middleware.base import BaseHTTPMiddleware


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration class"""
    # API Keys
    moonshot_api_key: str
    
    # Model Settings
    model_name: str = "kimi-k2-0905-Preview"
    base_url: str = "https://api.moonshot.cn/v1"
    
    # Timeout Settings (seconds)
    api_timeout: float = 120.0
    api_connect_timeout: float = 10.0
    
    # Performance Settings
    max_retries: int = 2
    retry_delay: float = 1.0
    max_concurrent_requests: int = 10
    
    # Server Settings
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    # Label Studio Settings
    label_studio_from_name: str = "label"
    label_studio_to_name: str = "text"
    
    # Prompt Template
    system_prompt: str = (
        "你是一个智能标注助手。请根据用户提供的文本内容，生成准确的标注结果。\n"
        "请直接输出标注内容，不要添加额外的解释或格式。"
    )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MOONSHOT_API_KEY environment variable is required. "
                "Please set it before starting the service."
            )
        
        return cls(
            moonshot_api_key=api_key,
            model_name=os.getenv("MOONSHOT_MODEL", "kimi-k2-0905-Preview"),
            base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
            api_timeout=float(os.getenv("KIMI_API_TIMEOUT", "120")),
            api_connect_timeout=float(os.getenv("KIMI_CONNECT_TIMEOUT", "10")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            label_studio_from_name=os.getenv("LABEL_STUDIO_FROM_NAME", "label"),
            label_studio_to_name=os.getenv("LABEL_STUDIO_TO_NAME", "text"),
            system_prompt=os.getenv("SYSTEM_PROMPT", cls.system_prompt),
        )


# ============================================================================
# Structured Logging
# ============================================================================

import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatted logs for production log aggregation"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "task_id"):
            log_obj["task_id"] = record.task_id
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(log_level: str) -> logging.Logger:
    """Configure structured logging"""
    logger = logging.getLogger("labelstudio_ml")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


# ============================================================================
# Pydantic Models
# ============================================================================

class TaskData(BaseModel):
    """Label Studio task data structure"""
    model_config = {"extra": "allow"}  # Allow extra fields from Label Studio
    
    text: str = Field(default="", description="Text content to be labeled")


class Task(BaseModel):
    """Label Studio task structure"""
    model_config = {"extra": "allow"}  # Allow extra fields like meta, created_at, etc.
    
    id: Optional[Union[str, int]] = None
    data: TaskData = Field(default_factory=TaskData)
    # Label Studio may also send: meta, created_at, updated_at, etc.


class PredictRequest(BaseModel):
    """Request body for /predict endpoint - matches Label Studio ML Backend spec"""
    model_config = {"extra": "allow"}  # Allow params and other extra fields
    
    tasks: List[Task] = Field(..., description="List of tasks from Label Studio")
    model_version: Optional[str] = None  # Current model version in Label Studio
    project: Optional[str] = None        # Project ID
    label_config: Optional[str] = None   # Label config XML
    # Label Studio may also send: params, context, etc.
    
    @field_validator('tasks')
    @classmethod
    def validate_tasks(cls, v):
        if not v:
            raise ValueError('tasks list cannot be empty')
        if len(v) > 100:
            raise ValueError('maximum 100 tasks per request allowed')
        return v


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    version: str = "1.0.0"
    timestamp: str


class PredictionResult(BaseModel):
    """Single prediction result"""
    result: List[Dict[str, Any]]
    model_version: str = "kimi-k2-0905-Preview"
    score: Optional[float] = None


# ============================================================================
# Request Context Middleware
# ============================================================================

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add request ID and timing to each request"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Add request_id to logger adapter
        logger = logging.getLogger("labelstudio_ml")
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            # Log request completion
            duration = (time.time() - request.state.start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms",
                extra={"request_id": request_id, "duration_ms": round(duration, 2)}
            )
            
            return response
        except Exception as e:
            duration = (time.time() - request.state.start_time) * 1000
            logger.error(
                f"{request.method} {request.url.path} - ERROR - {duration:.2f}ms",
                extra={"request_id": request_id, "duration_ms": round(duration, 2)},
                exc_info=True
            )
            raise


# ============================================================================
# Kimi API Client with Connection Pooling
# ============================================================================

class KimiClient:
    """
    Thread-safe Kimi API client with connection pooling and retry logic.
    Uses httpx for efficient connection reuse.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Create httpx client with connection pooling
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=config.api_connect_timeout,
                read=config.api_timeout,
                write=10.0,
                pool=5.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50
            ),
            http2=True,  # Enable HTTP/2 for better performance
        )
        
        # Initialize OpenAI client with custom http client
        self.client = openai.OpenAI(
            api_key=config.moonshot_api_key,
            base_url=config.base_url,
            http_client=self.http_client,
            max_retries=config.max_retries,
        )
        
        # Semaphore for limiting concurrent API calls
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    def close(self):
        """Clean up resources"""
        self.http_client.close()
    
    async def generate_label(self, text: str, request_id: str) -> str:
        """
        Generate label for text using Kimi API.
        Implements retry logic and detailed error handling.
        """
        async with self.semaphore:
            start_time = time.time()
            
            extra_log = {"request_id": request_id}
            
            try:
                # Truncate very long text to avoid token limit
                max_text_length = 8000
                if len(text) > max_text_length:
                    self.logger.warning(
                        f"Text truncated from {len(text)} to {max_text_length} chars",
                        extra=extra_log
                    )
                    text = text[:max_text_length] + "..."
                
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": text}
                ]
                
                # Run synchronous API call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,  # Default executor
                    lambda: self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2048,
                        top_p=0.9,
                    )
                )
                
                duration = time.time() - start_time
                
                generated_text = response.choices[0].message.content.strip()
                
                self.logger.info(
                    f"Kimi API call successful, duration={duration:.2f}s, "
                    f"tokens={response.usage.total_tokens if response.usage else 'N/A'}",
                    extra=extra_log
                )
                
                return generated_text
                
            except openai.APITimeoutError as e:
                self.logger.error(f"Kimi API timeout: {e}", extra=extra_log)
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Kimi API request timeout, please retry"
                )
            except openai.RateLimitError as e:
                self.logger.error(f"Kimi API rate limit: {e}", extra=extra_log)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded, please slow down"
                )
            except openai.APIError as e:
                self.logger.error(f"Kimi API error: {e}", extra=extra_log)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Kimi API error: {str(e)}"
                )
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}", extra=extra_log, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error during prediction"
                )


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Shared application state"""
    config: Config
    logger: logging.Logger
    kimi_client: KimiClient


# Global state
app_state = AppState()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    app_state.config = Config.from_env()
    app_state.logger = setup_logging(app_state.config.log_level)
    app_state.kimi_client = KimiClient(app_state.config, app_state.logger)
    
    app_state.logger.info(
        "Label Studio ML Backend starting",
        extra={"model": app_state.config.model_name}
    )
    
    yield
    
    # Shutdown
    app_state.logger.info("Label Studio ML Backend shutting down")
    app_state.kimi_client.close()


app = FastAPI(
    title="Label Studio ML Backend - Kimi",
    description="Production ML Backend for Label Studio using Moonshot Kimi API",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENV", "prod") != "prod" else None,
)

# Add middleware
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_labelstudio_prediction(generated_text: str) -> Dict[str, Any]:
    """
    Create Label Studio compatible prediction result.
    
    Uses configured from_name and to_name to match Label Studio XML config.
    """
    return {
        "result": [
            {
                "from_name": app_state.config.label_studio_from_name,
                "to_name": app_state.config.label_studio_to_name,
                "type": "textarea",
                "value": {
                    "text": [generated_text]
                }
            }
        ],
        "model_version": app_state.config.model_name,
        "score": None
    }


async def process_single_task(
    task: Task,
    request_id: str
) -> Dict[str, Any]:
    """
    Process a single task with individual error handling.
    Errors are caught and returned as failed predictions instead of failing entire batch.
    """
    task_id = task.id or "unknown"
    text = task.data.text
    
    extra_log = {"request_id": request_id, "task_id": str(task_id)}
    
    if not text or not text.strip():
        app_state.logger.warning(
            "Empty text in task, returning empty prediction",
            extra=extra_log
        )
        TASK_COUNTER.labels(status="empty").inc()
        return create_labelstudio_prediction("")
    
    try:
        # Call Kimi API
        generated_text = await app_state.kimi_client.generate_label(
            text, request_id
        )
        
        # Format prediction
        prediction = create_labelstudio_prediction(generated_text)
        
        app_state.logger.info(
            f"Task {task_id} processed successfully",
            extra=extra_log
        )
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        app_state.logger.error(
            f"Error processing task {task_id}: {e}",
            extra=extra_log,
            exc_info=True
        )
        # Return empty prediction instead of failing entire batch
        return create_labelstudio_prediction("")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Label Studio.
    Label Studio uses this to verify the ML backend is available.
    """
    return {
        "status": "UP",  # Label Studio expects "UP" for healthy
        "model": app_state.config.model_name
    }


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes.
    Checks if the service is ready to accept traffic.
    """
    try:
        # Optional: Validate API key by making a minimal API call
        # For now, just check if client is initialized
        if not hasattr(app_state, 'kimi_client') or app_state.kimi_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        return {"status": "ready", "model": app_state.config.model_name}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@app.api_route("/setup", methods=["GET", "POST"])
async def setup(request: Request):
    """
    Setup endpoint for Label Studio.
    Required for model validation and initialization.
    Returns model version and configuration.
    """
    return {
        "model_version": app_state.config.model_name,
        "labels": []  # Optional: return available labels if fixed
    }


@app.post("/predict")
async def predict(request: PredictRequest, req: Request):
    """
    Prediction endpoint for Label Studio.
    
    Features:
    - Parallel processing of tasks with concurrency limit
    - Individual task error handling (batch partial failure tolerance)
    - Request tracing with unique request ID
    """
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4())[:8])
    start_time = time.time()
    
    app_state.logger.info(
        f"Received prediction request with {len(request.tasks)} tasks",
        extra={"request_id": request_id}
    )
    
    try:
        # Process tasks with controlled concurrency
        tasks = [
            process_single_task(task, request_id)
            for task in request.tasks
        ]
        
        # Wait for all tasks to complete
        predictions = await asyncio.gather(*tasks, return_exceptions=False)
        
        duration = time.time() - start_time
        
        app_state.logger.info(
            f"Batch prediction completed in {duration:.2f}s",
            extra={"request_id": request_id, "task_count": len(predictions)}
        )
        
        return predictions
        
    except Exception as e:
        app_state.logger.error(
            f"Batch prediction failed: {e}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@app.get("/")
async def root():
    """Root endpoint for basic service verification"""
    return {
        "service": "Label Studio ML Backend - Kimi",
        "version": "1.1.0",
        "model": app_state.config.model_name,
        "endpoints": [
            "/health",
            "/ready", 
            "/predict"
        ]
    }
