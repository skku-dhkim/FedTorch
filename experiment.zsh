#! /bin/zsh

# Define a function to handle the interrupt signal
interrupt_handler() {
    echo "Script interrupted. Exiting..."
    kill -TERM 0
    exit 1
}

# Set the interrupt signal handler
trap interrupt_handler SIGINT


n_clients=20
sample_ratio=0.5
dirichlet_alpha=0.1
model="Simple_cnn"
dataset="Cifar-10"
local_iter=2
global_iter=2

methods=("FedAvg" "FedProx" "FedDyn" "FedNova" "FedBal")
#aggregators=("FedAvg" "Balancer")
aggregators=("FedAvg" "Balancer" "FedDE" "FedBF")
#aggregators=("FedDF" "FedBE")
#methods=("FedAvg")


for method in "${methods[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    exp_name="$1_${dirichlet_alpha}/${method}_${aggregator}"
    python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --aggregator $aggregator --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio
  done
done
