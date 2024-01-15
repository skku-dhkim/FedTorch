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
n_clients=10
sample_ratio=1.0
dirichlet_alpha=0.1
model="ConvNet"
dataset="Cifar-10"
gpu=True
gpu_frac=0.12
sigma=3
inverse=True
local_iter=10
global_iter=150

methods=("FedAvg" "FedProx" "FedNova" "FedDyn" "FedBal" )

for method in "${methods[@]}"
do
  aggregator="fedavg"
  exp_name="$1/${method}_${aggregator}"
  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma --inverse $inverse --cpus $cpus
done

for method in "${methods[@]}"
do
  aggregator="uniform"
  exp_name="$1/${method}_${aggregator}"
  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma --inverse $inverse --cpus $cpus
done

for method in "${methods[@]}"
do
  aggregator="balancer"
  exp_name="$1/${method}_${aggregator}"
  python run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --gpu $gpu --gpu_frac $gpu_frac --sample_ratio $sample_ratio --sigma $sigma --inverse $inverse --cpus $cpus
done
