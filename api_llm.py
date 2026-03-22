from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM API",
    description="API для работы с локальной LLM Ollama для агента",
    version="1.0.0"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"

class LLMRequest(BaseModel):
    prompt: str = Field(..., description="Текст запроса к LLM", min_length=1)
    temperature: float = Field(0.7, description="Креативность ответа (0-1)", ge=0, le=1)
    max_tokens: int = Field(256, description="Максимальная длина ответа", ge=1, le=1024)

class LLMResponse(BaseModel):
    prompt: str
    response: str
    model: str
    tokens_generated: int = None

@app.get("/")
async def root():
    return {
        "message": "LLM API for Country Expert Agent",
        "model": MODEL_NAME,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "ollama": "connected", "model": MODEL_NAME}
        else:
            return {"status": "degraded", "ollama": "error"}
    except:
        return {"status": "unhealthy", "ollama": "not_connected"}


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    try:
        logger.info(f"Запрос: {request.prompt[:100]}...")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ollama API error")

        result = response.json()

        return LLMResponse(
            prompt=request.prompt,
            response=result.get("response", ""),
            model=MODEL_NAME,
            tokens_generated=result.get("eval_count", 0)
        )

    except requests.exceptions.ConnectionError:
        logger.error("Не удалось подключиться к Ollama")
        raise HTTPException(status_code=503, detail="Ollama not available. Make sure Ollama is running.")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)