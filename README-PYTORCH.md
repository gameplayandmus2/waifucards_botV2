# Переключение между GPU и CPU версиями PyTorch

## Быстрое переключение

### Для GPU (clipCards.py - быстро):
```bash
.\install-gpu.bat
```

### Для CPU (bot.py на сервере):
```bash
.\install-cpu.bat
```

## Что делают эти скрипты?

1. **install-gpu.bat** - удаляет текущую версию PyTorch и ставит CUDA 12.4 версию
   - PyTorch 2.6.0+cu124
   - Использует RTX 4070 Ti SUPER
   - **Используй перед запуском clipCards.py**

2. **install-cpu.bat** - удаляет текущую версию PyTorch и ставит CPU версию
   - PyTorch 2.9.1 (CPU only)
   - Работает везде
   - **Используй для развертывания на сервере**

## Пример использования

```bash
# Хочу создать FAISS индекс с GPU
.\install-gpu.bat
python clipCards.py

# Готово, теперь переключаюсь обратно на CPU для bot.py
.\install-cpu.bat
python bot.py
```

## Проверка текущей версии

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.__version__)"
```

**Если выводит:**
- `CUDA: True, Version: 2.6.0+cu124` - GPU версия активна
- `CUDA: False, Version: 2.9.1` - CPU версия активна
