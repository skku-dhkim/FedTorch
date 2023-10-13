#! /bin/bash

# Define a function to handle the interrupt signal
interrupt_handler() {
    echo "Script interrupted. Exiting..."
    kill -TERM 0
    exit 1
}

# Set the interrupt signal handler
trap interrupt_handler SIGINT

cpus=4
n_clients=20
sample_ratio=0.2
dirichlet_alpha=0.1
model="ConvNet"
dataset="Cifar-10"
gpu=True
gpu_frac=0.12
sigma=3

local_iter=10
global_iter=130

#methods=("FedAvg" "FedProx" "Scaffold" "FedNova" "FedBal" )
#
#for method in "${methods[@]}"
#do
#  balancer=False
#  exp_name="$1/${method}_${balancer}"
#  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma
#
#  balancer=True
#  exp_name="$1/${method}_${balancer}"
#  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma
#done

method="FedBal"
balancer=True
inverse=False
exp_name="$1/${method}_INV_${balancer}"
python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma --inverse $inverse --cpus $cpus

#
#balancer=False
#for method in "${methods[@]}"
#do
#  exp_name="$1/${method}_${balancer}"
#  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio
#done
