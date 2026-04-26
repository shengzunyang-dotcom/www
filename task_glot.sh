
monitor_gpu_processes() {
    while true; do
        # 使用nvidia-smi命令获取指定GPU上正在运行的进程数量
        gpu_processes=$(ps -ef | grep train | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index 上有 $gpu_processes 个程序在运行，将睡眠3分钟..."
            sleep 60  # 睡眠3分钟
        else
            echo "GPU $gpu_index 上没有程序在运行，退出循环。"
            break
        fi
    done
}

export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.1-a100-0 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.1-a100-1 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.1-a100-2 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.1-a100-3 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.1-a100-4 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.1-a100-5 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.1-a100-6 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.1-a100-7 --domain_type 2 --lamda 0.1 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.2-a100-8 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.2-a100-9 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.2-a100-10 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.2-a100-11 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.2-a100-12 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.2-a100-13 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.2-a100-14 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.2-a100-15 --domain_type 2 --lamda 0.2 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.3-a100-16 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.3-a100-17 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.3-a100-18 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.3-a100-19 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.3-a100-20 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.3-a100-21 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.3-a100-22 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.3-a100-23 --domain_type 2 --lamda 0.3 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.4-a100-24 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.4-a100-25 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.4-a100-26 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.4-a100-27 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.4-a100-28 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.4-a100-29 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.4-a100-30 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.4-a100-31 --domain_type 2 --lamda 0.4 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.5-a100-32 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.5-a100-33 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.5-a100-34 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.5-a100-35 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.5-a100-36 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.5-a100-37 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.5-a100-38 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.5-a100-39 --domain_type 2 --lamda 0.5 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.6-a100-40 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.6-a100-41 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.6-a100-42 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.6-a100-43 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.6-a100-44 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.6-a100-45 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.6-a100-46 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.6-a100-47 --domain_type 2 --lamda 0.6 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.7-a100-48 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.7-a100-49 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.7-a100-50 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.7-a100-51 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.7-a100-52 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.7-a100-53 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.7-a100-54 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.7-a100-55 --domain_type 2 --lamda 0.7 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.8-a100-56 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.8-a100-57 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.8-a100-58 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.8-a100-59 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.8-a100-60 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.8-a100-61 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.8-a100-62 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.8-a100-63 --domain_type 2 --lamda 0.8 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.9-a100-64 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.9-a100-65 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.9-a100-66 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.9-a100-67 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.9-a100-68 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.9-a100-69 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.9-a100-70 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.9-a100-71 --domain_type 2 --lamda 0.9 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.10-a100-72 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.10-a100-73 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.10-a100-74 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.10-a100-75 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
export CUDA_VISIBLE_DEVICES=0; nohup python train.py --wandb_name mosei-0.10-a100-76 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python train.py --wandb_name mosei-0.10-a100-77 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python train.py --wandb_name mosei-0.10-a100-78 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python train.py --wandb_name mosei-0.10-a100-79 --domain_type 2 --lamda 0.10 > log/0.log 2>&1 &
sleep 30
monitor_gpu_processes
