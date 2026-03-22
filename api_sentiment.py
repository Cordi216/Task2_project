from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import uvicorn
from typing import List

app = FastAPI(
    title="Sentiment Analysis API",
    description="Анализ тональности текста - оценка отзывов о странах)",
    version="1.0.0"
)

print("Загрузка модели")
classifier = pipeline(
    "sentiment-analysis",
    model="cointegrated/rubert-tiny-sentiment-balanced"
)
print("Модель загрузилась")

class SentimentRequest(BaseModel):
    text: str = Field(..., description="Текст для анализа", min_length=1)

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="Список текстов для анализа")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

class StatsRequest(BaseModel):
    texts: List[str] = Field(..., description="Список отзывов для статистики")

class StatsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    neutral: int
    positive_percent: float
    negative_percent: float
    neutral_percent: float
    details: List[dict]

@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API for Country Expert Agent",
        "endpoints": {
            "predict": "POST /predict - анализ одного текста",
            "batch": "POST /batch - анализ нескольких текстов",
            "stats": "POST /stats - статистика по отзывам"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "rubert-tiny-sentiment-balanced"}


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        result = classifier(request.text)
        prediction = result[0]

        return SentimentResponse(
            text=request.text,
            label=prediction['label'],
            score=prediction['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=BatchSentimentResponse)
async def batch_predict(request: BatchSentimentRequest):
    try:
        results = []
        for text in request.texts:
            result = classifier(text)
            results.append(SentimentResponse(
                text=text,
                label=result[0]['label'],
                score=result[0]['score']
            ))
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stats", response_model=StatsResponse)
async def get_sentiment_stats(request: StatsRequest):
    try:
        labels = []
        details = []

        for text in request.texts:
            result = classifier(text)
            label = result[0]['label']
            score = result[0]['score']
            labels.append(label)
            details.append({
                "text": text[:100],
                "label": label,
                "score": round(score, 3)
            })

        total = len(labels)
        positive = labels.count("positive")
        negative = labels.count("negative")
        neutral = labels.count("neutral")

        return StatsResponse(
            total=total,
            positive=positive,
            negative=negative,
            neutral=neutral,
            positive_percent=round(positive / total * 100, 1) if total > 0 else 0,
            negative_percent=round(negative / total * 100, 1) if total > 0 else 0,
            neutral_percent=round(neutral / total * 100, 1) if total > 0 else 0,
            details=details
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)