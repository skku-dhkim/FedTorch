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
sample_ratio=0.2
dirichlet_alpha=0.1
model="resnet-18"
dataset="Cifar-10"
gpu=True
gpu_frac=0.16

local_iter=5
global_iter=50

methods=("FedAvg" "FedProx" "Scaffold" "FedNova" "FedBal" )

for method in "${methods[@]}"
do
  balancer=True
  exp_name="$1/${method}_${balancer}"
  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio

  balancer=False
  exp_name="$1/${method}_${balancer}"
  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio
done
#
#balancer=False
#for method in "${methods[@]}"
#do
#  exp_name="$1/${method}_${balancer}"
#  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio
#done
