#!/bin/bash

# 设置信号处理函数
trap 'echo "\n收到中断信号，正在退出..."; exit 0' SIGINT

# 无限循环执行run_GCN.py
while true; do
    echo "开始新一轮执行: $(date '+%Y-%m-%d %H:%M:%S')"
    python3 run_GCN.py
    echo "本轮执行结束: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-----------------------------------"
done