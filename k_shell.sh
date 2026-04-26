monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(ps -ef | grep mosi_hp | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 60  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}
export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 0.2 --alpha2 1 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 0.2 --alpha2 1 > log/test.log41 2>&1 &

export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 0.4 --alpha2 1 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 0.4 --alpha2 1 > log/test.log41 2>&1 &

export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 0.6 --alpha2 1 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 0.6 --alpha2 1 > log/test.log41 2>&1 &

sleep 30
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 0.8 --alpha2 1 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 0.8 --alpha2 1 > log/test.log41 2>&1 &

export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.2 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.2 > log/test.log41 2>&1 &

export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.4 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.4 > log/test.log41 2>&1 &

sleep 30
monitor_gpu_processes

export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.6 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.6 > log/test.log41 2>&1 &

export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.8 > log/test.log41 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --alpha1 1 --alpha2 0.8 > log/test.log41 2>&1 &


# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --abl_mode TA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --abl_mode TA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --abl_mode TA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --abl_mode TA > log/test.log41 2>&1 &

# sleep 30
# monitor_gpu_processes
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --abl_mode VA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --abl_mode VA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --abl_mode VA > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --abl_mode VA > log/test.log41 2>&1 &

# sleep 30
# monitor_gpu_processes
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 12 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 12 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 12 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 12 > log/test.log41 2>&1 &

# sleep 30
# monitor_gpu_processes
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 14 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 14 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 14 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 14 > log/test.log41 2>&1 &

# sleep 30
# monitor_gpu_processes
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 16 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=0; nohup python train_mosi_hp.py --topk 16 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 16 > log/test.log41 2>&1 &
# export CUDA_VISIBLE_DEVICES=1; nohup python train_mosi_hp.py --topk 16 > log/test.log41 2>&1 &