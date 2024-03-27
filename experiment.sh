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
sample_ratio=1.0
#dirichlet_alpha=0.05
#model="Simple_CNN"
#model="Resnet-18"
model="Vgg11"


dataset="Cifar-10"
gpu=True
gpu_frac=0.12
inverse=True
local_iter=10
global_iter=100
T=0.5
NT=2.0

methods=("FedAvg" "FedProx" "FedDyn" "FedNova" "FedBal")
#methods=("FedAvg")
aggregators=("FedAvg" "Balancer")
#aggregators=("FedAvg" "Balancer" "FedDF" "FedBE")
#aggregators=("FedDF" "FedBE")


dirichlet_alpha=0.05
for method in "${methods[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    exp_name="$1_${dirichlet_alpha}/${method}_${aggregator}"
    python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio --cpu $cpus --gpu $gpu --gpu_frac $gpu_frac --inverse $inverse --T $T --NT $NT
  done
done

dirichlet_alpha=0.1
for method in "${methods[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    exp_name="$1_${dirichlet_alpha}/${method}_${aggregator}"
    python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio --cpu $cpus --gpu $gpu --gpu_frac $gpu_frac --inverse $inverse --T $T --NT $NT
  done
done

dirichlet_alpha=100
for method in "${methods[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    exp_name="$1_${dirichlet_alpha}/${method}_${aggregator}"
    python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio --cpu $cpus --gpu $gpu --gpu_frac $gpu_frac --inverse $inverse --T $T --NT $NT
  done
done

