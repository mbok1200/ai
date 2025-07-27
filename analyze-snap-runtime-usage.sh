#!/bin/bash

echo "🔍 Збір інформації про Snap-пакети та їхні залежності..."

all_snaps=$(snap list | tail -n +2 | awk '{print $1}')
declare -A runtime_usage

for snap in $all_snaps; do
    used=$(snap connections "$snap" | grep content | awk '{print $3}' | cut -d: -f1)
    if [ -n "$used" ]; then
        for rt in $used; do
            runtime_usage["$rt"]+="$snap "
        done
    fi
done

echo ""
echo "📦 Використані рунтайми:"
for rt in "${!runtime_usage[@]}"; do
    echo "  ✅ $rt: ${runtime_usage[$rt]}"
done

echo ""
echo "🗑 Можливо невикористовувані рунтайми:"

# Візьмемо список рунтаймів серед усіх встановлених (gnome, kde, core, gtk, mesa, etc.)
all_runtimes=$(snap list | tail -n +2 | awk '{print $1}' | grep -E '^(gnome|kde|core|gtk|mesa|qt|flutter|gnome-|core|snapd)')

unused_found=0
for rt in $all_runtimes; do
    if [ -z "${runtime_usage[$rt]}" ]; then
        size=$(sudo du -sh /var/snap/"$rt" 2>/dev/null | cut -f1)
        echo "  ❌ $rt (можливо не використовується, ~ $size)"
        unused_found=1
    fi
done

if [ $unused_found -eq 0 ]; then
    echo "  ✅ Невикористовуваних рунтаймів не знайдено."
fi
