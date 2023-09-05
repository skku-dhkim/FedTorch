#! /bin/zsh

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
model="simple_cnn"
dataset="Cifar-10"

balancer=False
method="FedAvg"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedProx"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="Scaffold"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedNova"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedBal"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name


# With balancer experiment
balancer=True
method="FedAvg"
exp_name="$1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedProx"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="Scaffold"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedNova"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name

method="FedBal"
exp_name="Baseline1/${method}_${balancer}"
python3 run.py --n_clients $n_clients --dataset $dataset --dirichlet_alpha $dirichlet_alpha --method $method --model $model --exp_name $exp_name