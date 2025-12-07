# Развертывание бота на сервере

## Подготовка сервера

### 1. Установка зависимостей системы

```bash
# Обновление пакетов
sudo apt update && sudo apt upgrade -y

# Python и pip
sudo apt install python3 python3-pip python3-venv -y

# Библиотеки для OpenCV (важно для YOLO и обработки изображений)
# Для новых Ubuntu/Debian используй libgl1 вместо libgl1-mesa-glx
sudo apt install -y libgl1 libglib2.0-0 || sudo apt install -y libgl1-mesa-glx libglib2.0-0

# Утилиты
sudo apt install dos2unix git -y
```

### 2. Клонирование проекта

```bash
cd /var/www
git clone <your-repo-url> waifucards_botV2
cd waifucards_botV2
```

### 3. Создание виртуального окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Установка зависимостей (CPU версия)

```bash
# Исправь line endings если скопировал с Windows
dos2unix install-cpu.sh

# Дай права на выполнение
chmod +x install-cpu.sh

# Установи зависимости
./install-cpu.sh
```

**Альтернативный способ (без скрипта):**
```bash
pip install -r requirements.txt
```

### 5. Настройка .env файла

```bash
# Создай .env файл
nano .env
```

Добавь в него:
```env
TELEGRAM_TOKEN=your_token_here
USE_GPU=false
BASE_IMG_DIR=../goddess-story/static/img
```

**Примечание:** Если путь к картинкам на сервере другой, измени `BASE_IMG_DIR` на правильный путь.

### 6. Проверка данных

Убедись что есть необходимые файлы:
```bash
ls data/
# Должны быть: faiss.index, id_map.json, cards.json
```

Если их нет - создай индексы локально с GPU (на Windows) и загрузи на сервер.

### 7. Запуск бота

**Тестовый запуск:**
```bash
python3 bot.py
```

**Запуск в фоне (с помощью screen):**
```bash
# Установи screen если нет
sudo apt install screen -y

# Создай сессию
screen -S waifubot

# Активируй venv
source .venv/bin/activate

# Запусти бота
python3 bot.py

# Отключись от сессии: Ctrl+A, потом D
```

**Вернуться к боту:**
```bash
screen -r waifubot
```

**Список активных сессий:**
```bash
screen -ls
```

### 8. Автозапуск через systemd (опционально)

Создай сервис:
```bash
sudo nano /etc/systemd/system/waifubot.service
```

Вставь:
```ini
[Unit]
Description=Waifu Cards Telegram Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/waifucards_botV2
Environment="PATH=/var/www/waifucards_botV2/.venv/bin"
ExecStart=/var/www/waifucards_botV2/.venv/bin/python3 bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Активируй сервис:
```bash
sudo systemctl daemon-reload
sudo systemctl enable waifubot
sudo systemctl start waifubot
```

Проверь статус:
```bash
sudo systemctl status waifubot
```

Просмотр логов:
```bash
sudo journalctl -u waifubot -f
```

## Обновление на сервере

```bash
cd /var/www/waifucards_botV2

# Останови бота (если используешь systemd)
sudo systemctl stop waifubot

# Или останови screen сессию
screen -r waifubot
# Нажми Ctrl+C, потом exit

# Обнови код
git pull

# Активируй venv
source .venv/bin/activate

# Обнови зависимости если нужно
pip install -r requirements.txt

# Запусти снова
sudo systemctl start waifubot
# или
screen -S waifubot
source .venv/bin/activate
python3 bot.py
```

## Обновление индексов FAISS

Индексы нужно создавать локально на Windows с GPU, потому что это быстрее:

**На Windows:**
```bash
.\install-gpu.bat
python clipCards.py
```

**Загрузка на сервер:**
```bash
# С локальной машины
scp data/faiss.index root@server:/var/www/waifucards_botV2/data/
scp data/id_map.json root@server:/var/www/waifucards_botV2/data/
scp data/cards.json root@server:/var/www/waifucards_botV2/data/
```

**На сервере:**
```bash
# Перезапусти бота чтобы он подхватил новые индексы
sudo systemctl restart waifubot
```

## Проблемы и решения

### Line endings проблема

```bash
dos2unix install-cpu.sh install-gpu.sh
```

### ModuleNotFoundError

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Бот не отвечает

```bash
# Проверь что бот запущен
ps aux | grep bot.py

# Проверь логи
sudo journalctl -u waifubot -n 50
```

### Нет памяти

```bash
# Проверь использование памяти
free -h

# Уменьши количество workers в clipCards.py
# max_workers = 4  # вместо 16
```
