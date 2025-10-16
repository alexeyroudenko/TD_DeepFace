FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем переменные окружения для подавления предупреждений TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=-1
ENV PYTHONWARNINGS=ignore
ENV KMP_AFFINITY=noverbose

# Устанавливаем системные зависимости, необходимые для DeepFace
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    libgl1-mesa-dri \
    libgl1-mesa-dev \
    libegl-mesa0 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы требований
COPY requirements.txt .

# Устанавливаем зависимости без обновления pip (to avoid segfault)
RUN pip install --no-cache-dir -r requirements.txt

# Создаем директорию для весов моделей
RUN mkdir -p /root/.deepface/weights

# Копируем скрипт инициализации моделей
COPY init_models.py /tmp/init_models.py

# Note: Model initialization happens at runtime when weights are available from mounted volume

# Копируем код приложения
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]