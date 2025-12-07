#!/bin/bash

echo "========================================"
echo "Установка CPU версии PyTorch"
echo "========================================"
echo ""

pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

echo ""
echo "========================================"
echo "Проверка установки:"
echo "========================================"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo ""
echo "Готово! CPU версия установлена"
