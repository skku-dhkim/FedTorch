from src import *
from src.clients import *
from src.model import *
from torch.nn import Module
from src.utils.logger import write_experiment_summary


class Aggregator:
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 **kwargs):
        # Client Meta setting
        self.name = 'aggregator'

        # Data setting
        self.test_loader: DataLoader = test_data
        self.valid_loader: DataLoader = valid_data

        # Training settings
        self.training_settings = train_settings
        self.global_iter = 0
        self.lr = train_settings['global_lr']

        # Model
        self.num_of_classes = NUMBER_OF_CLASSES[dataset_name]
        self.model: Module = model_call(train_settings['model'], NUMBER_OF_CLASSES[dataset_name])

        main_dir = Path(log_path).parent.absolute()
        root_dir = Path("./logs").absolute()

        # NOTE: We will use same initial model for same experiment.
        if main_dir == root_dir:
            write_experiment_summary("Aggregator", {"Model": "Make new"})
            pass
        elif os.path.isfile(os.path.join(main_dir, 'init_model.pt')):
            weights = torch.load(os.path.join(main_dir, 'init_model.pt'))
            self.model.load_state_dict(weights)
            write_experiment_summary("Aggregator", {"Model": "Load from path"})
        else:
            torch.save(self.model.state_dict(), os.path.join(main_dir, 'init_model.pt'))
            write_experiment_summary("Aggregator", {"Model": "Make new and save into init_model.pt"})

        # Log path
        self.summary_path = os.path.join(log_path, "{}".format(self.name))
        self.summary_writer = SummaryWriter(os.path.join(self.summary_path, "summaries"))

        # Device setting
        self.device = "cuda" if train_settings['use_gpu'] is True else "cpu"
        # TODO: Device check is needed.
        # self.device = "cpu"
        self.model.to(self.device)

        # Federated learning method settings
        self.__collected_weights: Optional[dict] = None

        # ETC
        self.kwargs = kwargs

        # Initial model accuracy
        self.test_accuracy = self.compute_accuracy()
        self.best_acc = self.test_accuracy

        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)

    def set_parameters(self, state_dict: Union[OrderedDict, dict], strict=True) -> None:
        self.model.load_state_dict(state_dict, strict=strict)

    def get_parameters(self, ordict: bool = True) -> Union[OrderedDict, list]:
        if ordict:
            return OrderedDict({k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()})
        else:
            return [val.clone().detach().cpu() for _, val in self.model.state_dict().items()]

    def save_model(self):
        save_path = os.path.join(self.summary_path, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "Best_global_model.pt".format(self.global_iter))

        torch.save({
            'global_epoch': self.global_iter,
            'client_name': self.name,
            'model': self.model.state_dict()
        }, save_path)

    def save_data(self):
        save_path = os.path.join(self.summary_path, "test_data.pt")
        torch.save(self.test_loader, save_path)

    def compute_accuracy(self) -> float:
        """
        Returns:
            (float) training_acc: Training accuracy of client's test data.
        """
        correct = []
        total = []
        self.model.eval()
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

    def update_test_acc(self):
        # update test accuracy
        self.test_accuracy = self.compute_accuracy()
        self.summary_writer.add_scalar('global_test_acc', self.test_accuracy, self.global_iter)
        if self.test_accuracy > self.best_acc:
            self.best_acc = self.test_accuracy
            self.save_model()
            return True
        else:
            return False

    def calc_cos_similarity(self, original_state: OrderedDict, current_state: OrderedDict) -> Optional[OrderedDict]:
        result = OrderedDict()
        for k in current_state.keys():
            score = self.cos_sim(torch.flatten(original_state[k].to(torch.float32)),
                                 torch.flatten(current_state[k].to(torch.float32)))
            result[k] = score
            self.summary_writer.add_scalar("weight_similarity_before/{}".format(k), score, self.global_iter)
            self.summary_writer.add_scalar("weight_similarity_after/{}".format(k), score, self.global_iter)
        return result


class AvgAggregator(Aggregator):
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 **kwargs):

        super().__init__(test_data, valid_data, dataset_name, log_path, train_settings, **kwargs)
        self.test_accuracy = self.compute_accuracy()
        self.name = 'Average Aggregator'

    def fed_avg(self, clients: List[Client], global_lr: float, mode='fedavg'):
        empty_model = OrderedDict()
        total_len = sum(client.data_len() for client in clients)

        for k, v in self.model.state_dict().items():
            for client in clients:
                if mode == 'fedavg':
                    p = (client.data_len() / total_len)
                else:
                    p = 1 / len(clients)

                empty_model[k] = empty_model.get(k, 0) + client.model[k] * p * global_lr

        # Global model updates
        self.set_parameters(empty_model)
        self.global_iter += 1


# TODO: Aggregation balancer doesn't need any more. Check for deprecation.
class AggregationBalancer(Aggregator):
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 **kwargs):

        super().__init__(test_data, valid_data, dataset_name, log_path, train_settings, **kwargs)
        self.test_accuracy = self.compute_accuracy()

    def aggregation_balancer(self,
                             clients: List[Client],
                             global_lr: float = 1.0, temperature: float = 1.0, sigma: int = 1, inverse: bool = True):
        previous_g_model = self.model.state_dict()
        empty_model = OrderedDict((key, []) for key in self.model.state_dict().keys())

        # INFO: Collect the weights from all client in a same layer.
        for k, v in self.model.state_dict().items():
            for client in clients:
                empty_model[k].append(client.model[k])
            empty_model[k] = torch.stack(empty_model[k])

        importance_score = None
        for name, v in empty_model.items():
            if 'classifier' in name and 'weight' in name:
                # NOTE: FC layer for calculate importance.
                importance_score = self.client_importance_score(empty_model[name],
                                                                'cos',
                                                                previous_g_model[name],
                                                                sigma=sigma,
                                                                temperature=temperature, inverse=inverse)
                score = self.shape_convert(importance_score, name)
                empty_model[name] = torch.sum(score * v, dim=0) * global_lr
            elif 'classifier' in name and 'bias' in name:
                score = self.shape_convert(importance_score, name)
                empty_model[name] = torch.sum(score * v, dim=0) * global_lr
            else:
                # NOTE: Averaging the Feature extractor and others.
                empty_model[name] = torch.mean(v, 0) * global_lr

        # NOTE: Global model updates
        self.set_parameters(empty_model, strict=True)
        self.global_iter += 1

        # self.test_accuracy = self.compute_accuracy()
        # self.summary_writer.add_scalar('global_test_acc', self.test_accuracy, self.global_iter)
        self.summary_writer.add_histogram('aggregation_score', importance_score, self.global_iter)

    def client_importance_score(self, vector, method, global_model, normalize: bool = True, sigma=3, temperature=1.0,
                                inverse=True):
        weight_vec = vector.view(vector.size()[0], -1)

        if method == 'euclidean'.lower():
            g_vector = global_model.view(global_model.size()[0], -1).unsqueeze(0)
            # NOTE: Lower the distance less changes from global
            vector = torch.norm(g_vector - weight_vec, p=2, dim=-1)

            # NOTE: Make distance 0 if distance lower than standard deviation.
            std, mean = torch.std_mean(vector, dim=0)
            threshold = mean - sigma * std
            vector[vector < threshold] = 0

            # NOTE: Squeeze the dimension
            vector = vector.norm(p=2, dim=-1)
            score_vector = torch.exp(-vector)

            std, mean = torch.std_mean(score_vector, dim=0)
            threshold = mean + sigma * std
            score_vector[score_vector > threshold] = 0

        elif method == 'cos'.lower():
            g_vector = global_model.view(-1).unsqueeze(0)
            cos_similarity = torch.nn.CosineSimilarity(dim=-1)

            # NOTE: More similar large similarity value -> large similarity means small changes occurs from global model
            g_vector = g_vector.cpu()
            weight_vec = weight_vec.cpu()
            similarity = cos_similarity(g_vector, weight_vec)
            torch.nan_to_num_(similarity)

            # NOTE: Clipping the value if lower than threshold
            std, mean = torch.std_mean(similarity, dim=-1)
            threshold = mean - sigma * std
            similarity[similarity < threshold] = threshold

            # NOTE: Projection
            if inverse:
                # NOTE: Large similar (x=1) -> Has large weights
                score_vector = torch.exp(similarity)
            else:
                # NOTE: Large similar (x=1) -> Has less weights
                score_vector = torch.exp(-similarity)
        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

        if normalize:
            T = temperature
            score_vector = torch.softmax(score_vector / T, dim=0)
        return score_vector

    def shape_convert(self, score, layer):
        if 'bias' in layer:
            return score.unsqueeze(-1)
        if 'features' in layer:
            return score.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif 'classifier' in layer:
            return score.unsqueeze(-1).unsqueeze(-1)
        else:
            return score
