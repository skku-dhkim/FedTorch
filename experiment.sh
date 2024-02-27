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
dirichlet_alpha=0.05
model="ConvNet"
dataset="Cifar-10"
gpu=True
gpu_frac=0.12
inverse=True
local_iter=10
global_iter=200
T=0.5
NT=2.0

methods=("FedAvg" "FedProx" "FedDyn" "FedNova" "FedBal")
#aggregators=("FedAvg" "Balancer")
aggregators=("FedAvg" "Balancer" "FedDE" "FedBF")

for method in "${methods[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    exp_name="$1_${dirichlet_alpha}/${method}_${aggregator}"
    python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio --cpu $cpus --gpu $gpu --gpu_frac $gpu_frac --inverse $inverse --T $T --NT $NT
  done
done

