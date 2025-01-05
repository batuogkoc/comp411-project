srun -N 1 -n 4 --mem-per-cpu=16G --qos=users --partition=short -t 2:00:00 --gres=gpu:1 --constraint=tesla_v100 --pty bash
# srun -N 1 -n 4 --qos=users --partition=mid -t 03:00:00  --pty bash