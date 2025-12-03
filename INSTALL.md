# Инструкции по установке зависимостей

## Вариант 1: Локальная машина с GPU (быстрее)

Используйте это на машине с NVIDIA GPU (например, RTX 4070):

```bash
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

**Требования:**
- NVIDIA GPU с поддержкой CUDA
- NVIDIA CUDA Toolkit 12.4+ (будет автоматически загружен)
- Python 3.10+

**Преимущества:**
- Обработка карточек в 5-10x раз быстрее
- Рекомендуется для создания и обновления индекса

## Вариант 2: Сервер без GPU (универсально)

Используйте это на сервере или машине без GPU:

```bash
pip install -r requirements-cpu.txt
```

**Требования:**
- Python 3.10+
- Работает на любой системе (Linux, Windows, macOS)

**Особенности:**
- Медленнее, чем GPU версия (но все равно приемлемо)
- Универсальная версия, работает везде
- Рекомендуется для развертывания на сервере

## Как это работает

Скрипт `clipCards.py` автоматически определяет наличие GPU:

```python
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
```

Если установлена CUDA версия PyTorch и есть GPU - будет использована CUDA (быстро).
Если установлена CPU версия PyTorch - будет использован CPU (медленнее, но везде работает).

## Использование

**Локально с GPU:**
```bash
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu124
python clipCards.py
```

**На сервере без GPU:**
```bash
pip install -r requirements-cpu.txt
python clipCards.py
```

## Переключение между вариантами

Если вы установили CPU версию и хотите перейти на GPU:

```bash
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Если вы установили CUDA версию и хотите перейти на CPU:

```bash
pip install -r requirements-cpu.txt
```

## Проверка установки

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Version: {torch.__version__}')"
```

Если выводит `CUDA: True` - GPU работает.
Если выводит `CUDA: False` - используется CPU (это нормально для сервера).
