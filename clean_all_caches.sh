#!/bin/bash

set -e

echo "Починаємо очищення кешів pip, apt, snap, npm..."

# Функція для підтвердження
confirm() {
    read -rp "$1 [y/N]: " response
    [[ "$response" =~ ^[Yy]$ ]]
}

# 1. Очистити кеш pip
if confirm "Очистити кеш pip?"; then
    echo "Очищення кешу pip..."
    pip cache purge
    echo "Кеш pip очищено."
fi

# 2. Очистити кеш apt
if confirm "Очистити кеш apt (потрібен sudo)?"; then
    echo "Очищення кешу apt..."
    sudo apt clean
    echo "Автоматичне видалення непотрібних пакетів apt..."
    sudo apt autoremove -y
    echo "Кеш apt очищено."
fi

# 3. Очистити кеш snap
if confirm "Очистити кеш snap (потрібен sudo)?"; then
    echo "Очищення кешу snap..."
    sudo rm -rf /var/lib/snapd/cache/*
    echo "Кеш snap очищено."
fi

# 4. Очистити кеш npm
if command -v npm &>/dev/null; then
    if confirm "Очистити кеш npm?"; then
        echo "Очищення кешу npm..."
        npm cache clean --force
        echo "Кеш npm очищено."
    fi
else
    echo "npm не встановлено, пропускаємо очищення npm кешу."
fi

echo "Очищення кешів завершено!"
