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
sample_ratio=0.2
dirichlet_alpha=0.1
model="ConvNet"
dataset="Cifar-10"
local_iter=5
global_iter=100


methods=("FedAvg" "FedProx" "Scaffold" "FedNova" "FedBal" )
##methods=("FedAvg" "FedNova" "FedBal" )

balancer=True
for method in "${methods[@]}"
do
  exp_name="$1/${method}_${balancer}"
  python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio
done

balancer=False
for method in "${methods[@]}"
do
  exp_name="$1/${method}_${balancer}"
  python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name --balancer $balancer --local_iter $local_iter --global_iter $global_iter --sample_ratio $sample_ratio
done
