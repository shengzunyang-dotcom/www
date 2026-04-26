#!/bin/bash
# 自动化训练脚本 - 遍历学习率并重复10次
# 创建时间：$(date +%F)

# 配置参数
TRAIN_SCRIPT="train_mosei.py"  # 训练程序
REPEATS=20               # 重复次数
MIN_LR=1e-5              # 起始学习率
MAX_LR=5e-5              # 终止学习率
LR_STEPS=5               # 学习率步长数
LEARNING_RATES=(5e-6 8e-6 1e-5  2e-5  3e-5 4e-5)

echo "===== 开始自动化训练 ====="
echo "配置: $REPEATS 次重复, 学习率范围 $MIN_LR 到 $MAX_LR"

# 外层循环：重复实验
for ((rep=1; rep<=$REPEATS; rep++)); do
    echo -e "\n---- 第 $rep 次重复 ----"

    # 内层循环：遍历学习率
    for lr in "${LEARNING_RATES[@]}"; do
        printf "运行中: 学习率=%s\n" "$lr"
        
        # 执行训练程序
        python $TRAIN_SCRIPT --learning_rate "$lr"
        
        # 检查执行状态
        if [ $? -ne 0 ]; then
            echo "错误: 训练失败! 学习率=$lr" >&2
            exit 1
        fi
    done
done

echo -e "\n===== 所有任务完成 ====="