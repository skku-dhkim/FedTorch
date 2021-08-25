from dataset.data_loader import FedMNIST, FedCifar
from torch.utils.data import DataLoader
from torch import optim, nn
from clients.fed_clients import Client
from clients.fed_aggregator import Aggregator
from train.local_trainer import Trainer
from utils.logger import get_file_logger, get_stream_logger

# NOTE: Experiment meta
experiment_name = "baseline_test"
logging_path = "./logs/{}/experiment_summary.log".format(experiment_name)

# NOTE: Client settings
number_of_clients = 10
number_of_labels_per_client = 3
random_distribution = False

# NOTE: Data settings
dataset = "Cifar-10"
batch_size = 256
shuffle = True

# NOTE: Training parameters
model_name = "custom_CNN"
lr = 0.01
momentum = 0.9

# NOTE: Federated Settings
local_iter = 1
global_iter = 3

# NOTE: Logger Settings
stream_logger = get_stream_logger(__name__, "DEBUG")
file_logger = get_file_logger(__name__, logging_path, "INFO")

'''
MAIN Starting from here.
'''
# 1. Logging the setting meta.
file_logger.info("\n\t" + "*"*15 + " Client Settings " + "*"*15 +
                 "\n\t Number of clients: {}".format(number_of_clients) +
                 "\n\t Number of labels per client: {}".format(number_of_labels_per_client) +
                 "\n\t Random distribution: {}".format(random_distribution) +
                 "\n\t" + "*"*15 + " Training Hyper-parameters " + "*"*15 +
                 "\n\t Batch size: {}".format(batch_size) +
                 "\n\t Shuffle: {}".format(shuffle) +
                 "\n\t Learning rate: {}".format(lr) +
                 "\n\t Momentum: {}".format(momentum) +
                 "\n\t Local epochs: {}".format(local_iter) +
                 "\n\t Global iteration: {}".format(global_iter) +
                 "\n\t" + "*"*15 + " Dataset and Models " + "*"*15 +
                 "\n\t Dataset: {}".format(dataset) +
                 "\n\t Model: {}".format(model_name))

# 2. Loading the data
# TODO: Model should be called by name
fed_train, test, clients_meta = \
    FedCifar().load(number_of_clients, number_of_labels_per_client, random_dist=random_distribution)

file_logger.info("\n\t" + "*"*15 + " Client meta " + "*"*15 +
                 "\n\t" + str(clients_meta))

# 3. Client initialization
clients = []
for client in fed_train:
    _client = Client(client_name=client, train_data=fed_train[client])
    _client.train_loader = DataLoader(_client.dataset, batch_size=batch_size, shuffle=shuffle)
    clients.append(_client)

# 4. Create Aggregator
aggregator = Aggregator(model_name)
# model = model_manager.get_model("Resnet-50")
# model = model_manager.get_model("custom_CNN")


# 5. Create Trainer
trainer = Trainer(experiment_name=experiment_name)

# 6. Training Global Steps
for gr in range(global_iter):
    # 6-1: Assign global model to clients
    aggregator.model_assignment(clients)
    stream_logger.info("Global round: %d" % gr)

    # 6-2: Every clients train their model with their own dataset.
    for client in clients:
        stream_logger.debug("Working on client: %s" % client.name)
        # 6-3: Define loss function
        criterion = nn.CrossEntropyLoss()

        # 6-4: Define optimizer
        # optimizer = optim.Adam(client.model.parameters(), lr=lr)
        optimizer = optim.SGD(client.model.parameters(), lr=lr, momentum=momentum)

        # 6-5: Train steps
        trainer.train_steps(client, loss_fn=criterion, optimizer=optimizer, epochs=local_iter)

    # 6-6: FedAvg
    aggregator.fedAvg(clients)
    accuracy = aggregator.evaluation(test_data=test)
    stream_logger.info("Global accuracy: %2.2f %%" % accuracy)
    file_logger.info("[Global Round: {}/{}] Accuracy: {:2.2f}%".format(gr+1, global_iter, accuracy))
