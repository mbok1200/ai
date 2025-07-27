#!/bin/bash

echo "🔍 Пошук встановлених GNOME рунтаймів..."
all_gnome_snaps=$(snap list | awk '{print $1}' | grep -E '^gnome-')

echo ""
echo "📡 Аналіз підключень snap..."
used_gnome_snaps=$(snap connections | grep gnome | awk '{print $3}' | cut -d: -f1 | sort -u)

echo ""
echo "📦 Рунтайми GNOME, які використовуються:"
echo "$used_gnome_snaps" | sed 's/^/  ✅ /'

echo ""
echo "🧹 Невикористовувані GNOME рунтайми:"
unused=()
for snap in $all_gnome_snaps; do
    if ! echo "$used_gnome_snaps" | grep -qx "$snap"; then
        size=$(sudo du -sh /var/snap/"$snap" 2>/dev/null | cut -f1)
        echo "  🗑 $snap (прибл. $size)"
        unused+=("$snap")
    fi
done

echo ""
if [ ${#unused[@]} -eq 0 ]; then
    echo "✅ Усі GNOME рунтайми використовуються — нема що очищати."
    exit 0
fi

read -p "❓ Хочеш видалити ці ${#unused[@]} невикористовувані рунтайми? [y/N]: " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    for snap in "${unused[@]}"; do
        echo "➡️ Видалення $snap ..."
        sudo snap remove "$snap"
    done
    echo "✅ Готово!"
else
    echo "❎ Скасовано."
fi
