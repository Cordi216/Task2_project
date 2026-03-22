# country_agent.py
import requests

# API endpoints
SENTIMENT_API = "http://127.0.0.1:8000"
LLM_API = "http://127.0.0.1:8001"

# Примеры отзывов о странах (для демонстрации)
REVIEWS = {
    "япония": [
        "Невероятная культура и вежливые люди, обязательно вернусь",
        "Дорогой транспорт и маленькие отели, но это того стоит",
        "Сакура весной — это что-то невероятное",
        "Языковой барьер сложный, но locals помогают",
        "Фудзияма впечатляет, поездка запомнится надолго"
    ],
    "франция": [
        "Париж прекрасен, но очень много туристов",
        "Эйфелева башня впечатляет, очереди огромные",
        "Круассаны и сыры бесподобны",
        "Метро грязновато, будьте готовы",
        "Лувр — мечта, но одного дня мало"
    ],
    "италия": [
        "Пицца и паста — лучшие в мире",
        "Колизей впечатляет масштабом",
        "Очень жарко летом, сложно гулять",
        "Водители хаотичные, будьте осторожны",
        "Венеция уникальна, но дорого"
    ]
}


def ask_llm(prompt):
    """Запрос к LLM"""
    try:
        response = requests.post(
            f"{LLM_API}/generate",
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 200
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "Ошибка подключения к LLM"
    except Exception as e:
        return f"Ошибка: {e}"


def analyze_reviews(country):
    """Анализ отзывов о стране через Sentiment API"""
    country_lower = country.lower()
    if country_lower not in REVIEWS:
        return f"Нет данных об отзывах для {country}"

    try:
        response = requests.post(
            f"{SENTIMENT_API}/stats",
            json={"texts": REVIEWS[country_lower]}
        )

        if response.status_code == 200:
            data = response.json()
            return (f"Анализ отзывов о {country}:\n"
                    f"  Позитивных: {data['positive']} ({data['positive_percent']}%)\n"
                    f"  Нейтральных: {data['neutral']} ({data['neutral_percent']}%)\n"
                    f"  Негативных: {data['negative']} ({data['negative_percent']}%)")
        else:
            return "Ошибка при анализе отзывов"
    except Exception as e:
        return f"Ошибка подключения к Sentiment API: {e}"


def extract_country(text):
    """Извлекает название страны из текста (с учетом падежей)"""
    text_lower = text.lower()
    country_variants = {
        "япония": ["япония", "японии", "японию", "японией", "японии"],
        "франция": ["франция", "франции", "францию", "францией", "франции"],
        "италия": ["италия", "италии", "италию", "италией", "италии"]
    }

    for country, variants in country_variants.items():
        for variant in variants:
            if variant in text_lower:
                return country
    return None


def process_query(user_input):
    """Обработка запроса пользователя"""
    user_input_lower = user_input.lower()

    # Проверка: запрос об отзывах
    if "отзыв" in user_input_lower:
        country = extract_country(user_input)
        if country:
            return analyze_reviews(country)
        return "О какой стране хотите узнать отзывы? (Япония, Франция, Италия)"

    # Проверка: запрос о стране (есть название страны в запросе)
    country = extract_country(user_input)
    if country:
        # Передаем запрос пользователя в LLM как есть
        return ask_llm(user_input)

    # Любой другой запрос передаем в LLM
    return ask_llm(user_input)


def main():
    print("=" * 60)
    print("АГЕНТ ЭКСПЕРТ ПО СТРАНАМ")
    print("=" * 60)
    print("Спрашивайте о странах (Япония, Франция, Италия)")
    print("Можно спросить: 'Расскажи о Японии' или 'Какие отзывы о Японии?'")
    print("Введите 'exit' для выхода")
    print("=" * 60)

    while True:
        user_input = input("\nВы: ").strip()

        if user_input.lower() in ["exit", "quit", "выход"]:
            print("До свидания!")
            break

        if not user_input:
            continue

        print("\nАгент: ", end="")
        response = process_query(user_input)
        print(response)


if __name__ == "__main__":
    main()