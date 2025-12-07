#!/bin/bash

echo "========================================"
echo "Установка GPU версии PyTorch (CUDA 12.4)"
echo "========================================"
echo ""

pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "========================================"
echo "Проверка установки:"
echo "========================================"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo ""
echo "Готово! Теперь можешь запускать clipCards.py с GPU"
