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
                 valid_data: DataLoader,
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
        self.valid_loader = valid_data

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

        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)

    def set_parameters(self, state_dict: Union[OrderedDict, dict]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ordict: bool = True) -> Union[OrderedDict, Any]:
        if ordict:
            return OrderedDict({k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()})
        else:
            return [val.clone().detach().cpu() for _, val in self.model.state_dict().items()]

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
        self.model.to(self.device)
        optim = self.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.training_settings['local_lr'],
                               momentum=self.training_settings['momentum'],
                               weight_decay=1e-5)

        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        original_state = self.get_parameters()
        original_rep = self.compute_representations()

        # INFO: Local training logic
        for _ in range(self.training_settings['local_epochs']):
            training_loss = 0
            _summary_counter = 0

            # INFO: Training steps
            for x, y in self.train_loader:
                inputs = x
                labels = y

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                self.model.to(self.device)

                optim.zero_grad()

                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optim.step()

                # Summary Loss
                training_loss += loss.item()

                self.step_counter += 1
                _summary_counter += 1

                if _summary_counter % self.summary_count == 0:
                    training_acc = self.compute_accuracy()

                    self.summary_writer.add_scalar('training_loss', training_loss / _summary_counter, self.step_counter)
                    self.summary_writer.add_scalar('training_acc', training_acc, self.step_counter)
                    _summary_counter = 0

        # INFO: representation similarity with before global rep
        current_rep = self.compute_representations()
        self.calc_rep_similarity(original_rep, current_rep)
        self.cal_cos_similarity(original_state, self.get_parameters())

        self.save_model()
        self.global_iter += 1

    def cal_cos_similarity(self, original_state: OrderedDict, current_state: OrderedDict) -> Optional[OrderedDict]:
        result = OrderedDict()
        for k in current_state.keys():
            score = self.cos_sim(torch.flatten(original_state[k].to(torch.float32)),
                                 torch.flatten(current_state[k].to(torch.float32)))
            result[k] = score
            self.summary_writer.add_scalar("COS_similarity/{}".format(k), score, self.global_iter)
        return result

    def compute_accuracy(self) -> float:
        """
        Compute the accuracy using its test dataloader.
        Returns:
            (float) training_acc: Training accuracy of client's test data.
        """
        self.model.to(self.device)
        self.model.eval()

        correct = []
        total = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                y_max_scores, y_max_idx = outputs.max(dim=1)
                correct.append((y == y_max_idx).sum().item())
                total.append(len(x))
            training_acc = sum(correct) / sum(total)
        return training_acc

    def compute_representations(self) -> float:
        """
        Returns:
            (float) training_acc: Training accuracy of client's test data.
        """
        representations = {}
        rep = torch.tensor([]).to(self.device)

        # INFO: HELPER FUNCTION FOR FEATURE EXTRACTION
        def get_representations(name):
            def hook(model, input, output):
                representations[name] = output.detach()

            return hook

        # INFO: REGISTER HOOK
        # TODO: Model specific, need to make general for the future.
        self.model.features.register_forward_hook(get_representations('rep'))

        self.model.eval()
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                _ = self.model(x)
                rep = torch.cat([rep, representations['rep'].reshape((x.shape[0], -1))], dim=0)
        return rep

    def calc_rep_similarity(self, original_rep, current_rep) -> torch.Tensor:
        rd = self.cos_sim(original_rep, current_rep).cpu()
        score = torch.mean(rd)
        self.summary_writer.add_scalar("representations_similarity", score, self.global_iter)
        return score
