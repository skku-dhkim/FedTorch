import torch

from src import *
from src.model import *
from src.clients import *
from torch.nn import CrossEntropyLoss


@ray.remote
class Client:
    def __init__(self,
                 client_name: str,
                 dataset_name: str,
                 data: Optional[dict],
                 train_settings: dict,
                 log_path: str):
        # Client Meta setting
        self.name = client_name

        # Data settings
        self.train = data['train']
        self.train_loader = DataLoader(self.train,
                                       batch_size=train_settings['batch_size'], shuffle=True)

        self.test = data['valid']
        self.test_loader = DataLoader(self.test,
                                      batch_size=train_settings['batch_size'], shuffle=False)

        # Training settings
        self.training_settings = train_settings
        self.global_iter = 0
        self.step_counter = 0

        # Model
        self.model: FederatedModel = model_call(train_settings['model'], NUMBER_OF_CLASSES[dataset_name])

        # Optimizer
        self.optimizer = call_optimizer(train_settings['optim'])

        # Loss function
        self.loss = CrossEntropyLoss()

        # Log path
        self.summary_path = os.path.join(log_path, "{}".format(client_name))
        self.summary_writer = SummaryWriter(os.path.join(self.summary_path, "summaries"))
        self.summary_count = train_settings['summary_count']

        # Device setting
        self.device = "cuda" if train_settings['use_gpu'] is True else "cpu"

        # Save information
        self.save_data()
        self.save_hyper_parameters()

    def set_parameters(self, state_dict: Union[OrderedDict, dict]) -> None:
        self.model.set_parameters(state_dict)

    def get_parameters(self, ordict: bool = True) -> Union[OrderedDict, list]:
        return self.model.get_parameters(ordict)

    def save_model(self):
        save_path = os.path.join(self.summary_path, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "global_iter_{}.pt".format(self.global_iter))

        torch.save({
            'global_epoch': self.global_iter,
            'client_name': self.name,
            'model': self.model.state_dict()
        }, save_path)

    def save_data(self):
        save_path = os.path.join(self.summary_path, "client_data.pt")
        torch.save({
            'train': self.train,
            'test': self.test
        }, save_path)

    def save_hyper_parameters(self):
        save_path = os.path.join(self.summary_path, "hyper_parameter.txt")
        with open(save_path, "w") as file:
            file.write("Hyper-parameter of \'{}\'\n".format(self.name))
            for k, v in self.training_settings.items():
                file.write("{}: {}\n".format(k, v))

    def data_len(self):
        return len(self.train)

    async def train(self) -> None:
        optim = self.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.training_settings['local_lr'],
                               momentum=self.training_settings['momentum'],
                               weight_decay=1e-5)

        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        # Log of train and test set accuracy before training.
        # NOTE: You can make it to comments if you don't need logs
        # train_acc = self.compute_accuracy(data_loader=client.train_loader)
        # print("Train ACC Before training - {}: {:.2f}".format(client.name, train_acc))
        # test_acc = self.compute_accuracy(data_loader=self.test_loader)
        # print("Test ACC Before training - {}: {:.2f}".format(client.name, test_acc))

        # INFO: Local training logic
        # step_counter = 0
        for _ in range(self.training_settings['local_epochs']):
            training_loss = 0
            _summary_counter = 0

            # INFO: Training steps
            for x, y in self.train_loader:
                inputs = x
                labels = y

                inputs.to(self.device)
                labels.to(self.device)

                optim.zero_grad()

                outputs = self.model(inputs).to(self.device)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optim.step()

                # Summary Loss
                training_loss += loss.item()

                # step_counter += 1
                self.step_counter += 1
                _summary_counter += 1
                if _summary_counter % self.summary_count == 0:
                    training_acc = self.compute_accuracy()

                    self.summary_writer.add_scalar('training_loss',
                                                   training_loss / _summary_counter, self.step_counter)
                    self.summary_writer.add_scalar('training_acc',
                                                   training_acc, self.step_counter)

                    _summary_counter = 0

        # Log train and test set accuracy before training.
        # NOTE: You can make it to comments if you don't need logs
        # train_acc = self.compute_accuracy(data_loader=client.train_loader)
        # print("Train ACC After training - {}: {:.2f}".format(client.name, train_acc))
        # test_acc = self.compute_accuracy(data_loader=self.test_loader)
        # print("Test ACC After training - {}: {:.2f}".format(client.name, test_acc))
        self.save_model()
        self.global_iter += 1

    def compute_accuracy(self) -> float:
        """
        Compute the accuracy using its test dataloader.
        Returns:
            (float) training_acc: Training accuracy of client's test data.
        """

        correct = []
        total = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x).to(self.device)
                y_max_scores, y_max_idx = outputs.max(dim=1)
                correct.append((y == y_max_idx).sum().item())
                total.append(len(x))
            training_acc = sum(correct) / sum(total)
        return training_acc
