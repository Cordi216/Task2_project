import requests
import time

MODELS = ["qwen2.5:1.5b", "deepseek-r1:1.5b", "phi4-mini:3.8b"]

QUESTIONS = [
    "Расскажи кратко о Японии",
    "Назови столицу Франции",
    "Какая страна самая большая по площади?"
]

def test_model(model, question):
    url = "http://localhost:11434/api/generate"

    start = time.time()
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": question,
                "stream": False,
                "options": {"num_predict": 150}
            },
            timeout=60
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response": result.get("response", ""),
                "time": round(elapsed, 2),
                "tokens": result.get("eval_count", 0)
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


print("=" * 60)
print("СРАВНЕНИЕ LLM")
print("=" * 60)

results = {model: [] for model in MODELS}

for model in MODELS:
    print(f"\nМодель: {model}")
    print("-" * 40)

    for q in QUESTIONS:
        print(f"\nЗапрос: {q}")
        result = test_model(model, q)

        if result["success"]:
            print(f"Время: {result['time']} сек | Токенов: {result['tokens']}")
            print(f"Ответ: {result['response'][:70]}...")
            results[model].append({
                "question": q,
                "response": result["response"],
                "time": result["time"],
                "tokens": result["tokens"]
            })
        else:
            print(f"Ошибка: {result['error']}")
            results[model].append({
                "question": q,
                "error": result["error"]
            })

    print()


print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА")
print("=" * 60)
print(f"{'Модель':<18} {'Ср. время (сек)':<15} {'Ср. токенов':<12}")
print("-" * 60)

for model in MODELS:
    successful = [r for r in results[model] if "error" not in r]
    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_tokens = sum(r["tokens"] for r in successful) / len(successful)
        print(f"{model:<18} {avg_time:<15.2f} {avg_tokens:<12.0f}")
    else:
        print(f"{model:<18} {'Ошибка':<15} {'Ошибка':<12}")