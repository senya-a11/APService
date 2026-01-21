# Dockerfile (оптимизированная версия)
FROM python:3.11-slim as builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только requirements.txt для кэширования
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.11-slim

# Устанавливаем системные зависимости только для runtime
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости из builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY . .

# Создаем необходимые директории
RUN mkdir -p static/images static/uploads static/favicon data

# Создаем пользователя без привилегий (для безопасности)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Настройки
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]