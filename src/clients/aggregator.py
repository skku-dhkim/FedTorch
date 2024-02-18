import math

from src import *
from src.clients import *
from src.model import *
from torch.nn import Module
from src.train.train_utils import compute_layer_norms
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
        self.model: Module = model_call(train_settings['model'],
                                        NUMBER_OF_CLASSES[dataset_name],
                                        data_type=dataset_name.lower())
        self.client_model: Module = model_call(train_settings['model'],
                                               NUMBER_OF_CLASSES[dataset_name],
                                               data_type=dataset_name.lower())

        self.df_norm_diff = None
        self.previous_norm = None
        self.norm_gradient = None
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

    def measure_model_norm(self, measure_type, model=None):
        if model is None:
            model = self.model

        if measure_type == 'features':
            params = model.features.parameters()
        elif measure_type == 'classifier':
            params = model.classifier.parameters()
        else:
            params = model.parameters()

        vec = torch.cat([p.detach().view(-1) for p in params])
        l2_norm = torch.norm(vec, 2)
        return l2_norm

    def gradient_changes(self, previous_model, client_model):
        return previous_model - client_model

    def mark_model_diff(self, clients):
        prev_model_norm_all = self.measure_model_norm("all")

        if self.df_norm_diff is None:
            column_name = [str(client.name) for client in clients]
            self.df_norm_diff = pd.DataFrame(columns=column_name)

        tmp_dict = {}
        for client in clients:
            self.client_model.load_state_dict(client.model)
            client_model_norm = self.measure_model_norm("all", model=self.client_model)
            changes = self.gradient_changes(prev_model_norm_all, client_model_norm)
            self.summary_writer.add_scalar("weight norm/{}/{}".format(client.name, "all"),
                                           changes,
                                           self.global_iter)
            tmp_dict[str(client.name)] = changes.item()

        new_row = pd.DataFrame(tmp_dict, index=[0])
        self.df_norm_diff = pd.concat([self.df_norm_diff, new_row], ignore_index=True)


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
        # self.name = 'Average Aggregator'

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

        self.mark_model_diff(clients)

        # Global model updates
        self.set_parameters(empty_model)
        self.global_iter += 1


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
        self.importance_score: dict = {}

    def aggregation_balancer(self,
                             clients: List[Client],
                             global_lr: float = 1.0, temperature: float = 1.0, inverse: bool = True):
        previous_g_model = self.model.state_dict()
        empty_model = OrderedDict((key, []) for key in self.model.state_dict().keys())

        self.mark_model_diff(clients)

        # INFO: Normalize the model norm.
        server_norm = compute_layer_norms(self.model)
        minima_norm, alpha_range = self.find_min_and_max_norms(clients, server_norm)
        clients = self.weight_normalization(clients, minima_norm, alpha_range)

        # INFO: Collect the weights from all client in a same layer.
        for k, v in self.model.state_dict().items():
            if k not in self.importance_score.keys() and 'weight' in k:
                column_name = [str(client.name) for client in clients]
                self.importance_score[k] = pd.DataFrame(columns=column_name)

            for client in clients:
                empty_model[k].append(client.model[k])
            empty_model[k] = torch.stack(empty_model[k])

        # INFO: Aggregation with cosine angle differences.
        score_dict = defaultdict(lambda: defaultdict(float))
        for name, v in empty_model.items():
            if 'layer' not in name and 'fc' not in name:
                # NOTE: Averaging the Feature extractor(for very first layer of ResNet) and others.
                empty_model[name] = torch.mean(v, 0) * global_lr
            else:
                if 'weight' in name:
                    importance_score = self.client_importance_score(empty_model[name],
                                                                    previous_g_model[name],
                                                                    temperature=temperature,
                                                                    inverse=inverse)
                    score = self.shape_convert(importance_score, name)
                    empty_model[name] = torch.sum(score * v, dim=0) * global_lr

                    for i, client in enumerate(clients):
                        score_dict[name][str(client.name)] = importance_score[i].item()

                else:
                    score = self.shape_convert(importance_score, name)
                    empty_model[name] = torch.sum(score * v, dim=0) * global_lr
                    self.summary_writer.add_histogram('aggregation_score/{}'.format(name.split('.')[0]),
                                                      importance_score, self.global_iter+1)

        # NOTE: Global model updates
        self.set_parameters(empty_model, strict=True)
        self.global_iter += 1

        for k, v in score_dict.items():
            new_row = pd.DataFrame(v, index=[0])
            self.importance_score[k] = pd.concat([self.importance_score[k], new_row], ignore_index=True)

    def find_min_and_max_norms(self, clients: List, server_norms: dict):
        """
        Find the minimum norm from clients who participating the in training.
        Args:
            clients: (List) Clients list
            server_norms: (dict) global_t-1 norm
        Returns: (dict, dict) Minimal norm dictionary and maximum alpha range dictionary.
        """
        client_norms = [client.model_norm for client in clients]
        # List of clients norm calculated layer by layer.
        min_norms = {}
        max_norms = {}
        for norms in client_norms:
            for layer, norm in norms.items():
                _min_candidate = min(min_norms.get(layer, float('inf')), norm)
                max_norms[layer] = max(max_norms.get(layer, float('-inf')), norm)
                if _min_candidate <= server_norms[layer] and 'weight' in layer:
                    # Norm constrained should larger than global model
                    continue
                min_norms[layer] = _min_candidate
        alpha_range = {}
        for layer, norm in max_norms.items():
            alpha_range[layer] = norm - min_norms[layer]
        return min_norms, alpha_range

    def weight_normalization(self, clients: List[Client], minima_norm: dict, alpha_range: dict):
        """
        Normalize the weight of client's model. Alpha will adjust the minimum and maximum norm value.
        Args:
            clients: (List) Clients who participating the training
            minima_norm: (dict) minima norm of layers
            alpha_range: (dict) Adjust range value of each layers
        Returns: (List) Weight normalized clients list

        """
        for client in clients:
            for layer, param in client.model.items():
                alpha = self.calculate_alpha(alpha_range[layer])
                _current_norm = param.norm(p=2)
                layer_name = layer
                scaled_param = (param/_current_norm) * (minima_norm[layer_name]+alpha)
                param.copy_(scaled_param)
        return clients

    def calculate_alpha(self, max_alpha: float, temperature: float = 2.0):
        """
        Calculate the adjust alpha factor by global norm gradient.
        Alpha is calculated by 'alpha = exp(-T(x+1))'
        Args:
            max_alpha: (dict) Maximum value of alpha range.
            temperature: (float) Temperature factor that inversely proportional to gradient value
        Returns: (float) Adjustment alpha factor
        """
        if self.norm_gradient is None:
            # For the first iteration, we use initial alpha value.
            return max_alpha * math.exp(-temperature)
        pi_over_2 = math.pi / 2
        normalized_angle = self.norm_gradient/pi_over_2
        alpha = max_alpha * math.exp(-temperature*(normalized_angle+1))
        return alpha

    def client_importance_score(self, vector, global_model, normalize: bool = True, temperature=1.0, inverse=True):
        weight_vec = vector.view(vector.size()[0], -1)
        g_vector = global_model.view(-1).unsqueeze(0)
        cos_similarity = torch.nn.CosineSimilarity(dim=-1)

        g_vector = g_vector.cpu()
        weight_vec = weight_vec.cpu()
        similarity = cos_similarity(g_vector, weight_vec)
        torch.nan_to_num_(similarity)

        # NOTE: More similar large similarity value -> large similarity means small changes occurs from global model
        if inverse:
            score_vector = torch.exp(-similarity)
        else:
            score_vector = torch.exp(similarity)

        if normalize:
            score_vector = torch.softmax(score_vector / temperature, dim=0)

        return score_vector

    def shape_convert(self, score, layer):
        if 'bias' in layer:
            return score.unsqueeze(-1)
        if 'layer' in layer:
            return score.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif 'fc' in layer:
            return score.unsqueeze(-1).unsqueeze(-1)
        else:
            return score
