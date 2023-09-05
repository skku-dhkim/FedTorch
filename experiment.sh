#! /bin/bash

# Define a function to handle the interrupt signal
interrupt_handler() {
    echo "Script interrupted. Exiting..."
    kill -TERM 0
    exit 1
}

# Set the interrupt signal handler
trap interrupt_handler SIGINT


n_clients=10
dirichlet_alpha=0.1
model="resnet-18"
dataset="Cifar-10"
gpu=True
gpu_frac=0.16

balancer=True
method="FedAvg"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedProx"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="Scaffold"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedNova"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedBal"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac


# With balancer experiment
balancer=False
method="FedAvg"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedProx"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="Scaffold"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedNova"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac

method="FedBal"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --gpu $gpu --gpu_frac $gpu_frac
