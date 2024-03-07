import math

from src import *
from src.clients import *
from src.model import *
from torch.nn import Module
from src.train.train_utils import compute_layer_norms
from src.utils.logger import write_experiment_summary


def divergence(student_logits, teacher_logits):
    _divergence = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return _divergence


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
        self.previous_layer_norm = None
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

    def measure_model_norm(self, model=None):
        if model is None:
            model = self.model
        if type(model) is OrderedDict:
            vec = torch.cat([p.detach().view(-1) for p in model.values()])
        else:
            params = model.parameters()
            vec = torch.cat([p.detach().view(-1) for p in params])
        l2_norm = torch.norm(vec, 2)
        return l2_norm

    def measure_layer_norm_changed(self, current_norm):
        layer_grad = {}
        for (k1, current), (k2, previous) in zip(current_norm.items(), self.previous_layer_norm.items()):
            layer_grad[k1] = math.atan(current - previous)
        return layer_grad

    def mark_model_diff(self, clients):
        prev_model_norm_all = self.measure_model_norm()

        if self.df_norm_diff is None:
            column_name = [str(client.name) for client in clients]
            self.df_norm_diff = pd.DataFrame(columns=column_name)

        tmp_dict = {}
        for client in clients:
            client_model_norm = self.measure_model_norm(client.model)
            changes = prev_model_norm_all - client_model_norm
            self.summary_writer.add_scalar("weight norm/{}".format(client.name), changes, self.global_iter)
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

    def fed_avg(self, clients: List[Client], mode='fedavg'):
        empty_model = OrderedDict()
        total_len = sum(client.data_len() for client in clients)

        for k, v in self.model.state_dict().items():
            for client in clients:
                if mode == 'fedavg':
                    p = (client.data_len() / total_len)
                else:
                    p = 1 / len(clients)

                empty_model[k] = empty_model.get(k, 0) + client.model[k] * p * self.training_settings['global_lr']

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

    def aggregation_balancer(self, clients: List[Client]):
        previous_g_model = self.model.state_dict()
        empty_model = OrderedDict((key, []) for key in self.model.state_dict().keys())

        self.mark_model_diff(clients)

        # INFO: Normalize the model norm.
        minima_norm, alpha_range = self.find_min_and_max_norms(clients)
        clients = self.weight_normalization(clients, minima_norm, alpha_range)

        # INFO: Collect the weights from all client in a same layer.
        for k, v in self.model.state_dict().items():
            # NOTE: This lines are ONLY for the metric.
            if k not in self.importance_score.keys() and 'weight' in k:
                column_name = [str(client.name) for client in clients]
                self.importance_score[k] = pd.DataFrame(columns=column_name)

            for client in clients:
                empty_model[k].append(client.model[k])
            empty_model[k] = torch.stack(empty_model[k])

        # INFO: Aggregation with cosine angle differences.
        score_dict = defaultdict(lambda: defaultdict(float))
        for name, v in empty_model.items():
            if 'weight' in name:
                importance_score = self.client_importance_score(empty_model[name], previous_g_model[name])
                empty_model[name] = self.weight_apply(importance_score, v)

                for i, client in enumerate(clients):
                    score_dict[name][str(client.name)] = importance_score[i].item()

            else:       # Bias
                empty_model[name] = self.weight_apply(importance_score, v)
                self.summary_writer.add_histogram('aggregation_score/{}'.format(name.split('.')[0]),
                                                  importance_score, self.global_iter + 1)

        # NOTE: Global model updates
        self.set_parameters(empty_model, strict=True)
        self.global_iter += 1

        for k, v in score_dict.items():
            new_row = pd.DataFrame(v, index=[0])
            self.importance_score[k] = pd.concat([self.importance_score[k], new_row], ignore_index=True)

    def find_min_and_max_norms(self, clients: List):
        """
        Find the minimum norm from clients who participating the in training.
        Args:
            clients: (List) Clients list
        Returns: (dict, dict) Minimal norm dictionary and maximum alpha range dictionary.
        """
        server_norms = compute_layer_norms(self.model)
        client_norms = [client.model_norm for client in clients]
        # List of clients norm calculated layer by layer.
        min_norms = {}
        max_norms = {}
        for norms in client_norms:
            for layer, norm in norms.items():
                _min_candidate = min(min_norms.get(layer, float('inf')), norm)
                max_norms[layer] = max(max_norms.get(layer, float('-inf')), norm)
                if _min_candidate <= server_norms[layer]:
                    # Norm constrained should greater or equal than the global model
                    min_norms[layer] = server_norms[layer]
                else:
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
            for layer_name, param in client.model.items():
                if 'weight' in layer_name:  # Weights
                    alpha = self.calculate_alpha(alpha_range[layer_name],
                                                 layer_name,
                                                 temperature=self.training_settings['NT'])
                    _current_norm = param.data.norm(p=2).item()
                    scaled_param = (param / _current_norm) * (minima_norm[layer_name] + alpha)
                    param.copy_(scaled_param)
                else:  # Biases
                    _current_norm = param.norm(p=2)
                    alpha = self.calculate_alpha(alpha_range[layer_name],
                                                 layer_name,
                                                 temperature=self.training_settings['NT'])
                    scaled_param = (param / _current_norm) * (minima_norm[layer_name] + alpha)
                    param.copy_(scaled_param)
        return clients

    def calculate_alpha(self, max_alpha: float, weight_name: str, temperature: float = 2.0):
        """
        Calculate the adjust alpha factor by global norm gradient.
        Alpha is calculated by 'alpha = exp(-T(x+1))'
        Args:
            max_alpha: (dict) Maximum value of alpha range.
            weight_name: (str) Filter or FC layer weight name
            temperature: (float) Temperature factor that inversely proportional to gradient value
        Returns: (float) Adjustment alpha factor
        """
        if self.norm_gradient is None:
            # For the first iteration, we use initial alpha value.
            return max_alpha * 0.5
        pi_over_2 = math.pi / 2
        normalized_angle = self.norm_gradient[weight_name] / pi_over_2
        alpha = max_alpha * math.exp(-temperature * (normalized_angle + 1))
        # alpha = max_alpha * -0.5*(normalized_angle-1)                 # Inversely linear mapping
        return alpha

    def client_importance_score(self, vector, global_model):
        weight_vec = vector.view(vector.size()[0], -1)
        g_vector = global_model.view(-1).unsqueeze(0)
        cos_similarity = torch.nn.CosineSimilarity(dim=-1)

        g_vector = g_vector.cpu()
        weight_vec = weight_vec.cpu()
        similarity = cos_similarity(g_vector, weight_vec)
        torch.nan_to_num_(similarity)

        # NOTE: More similar large similarity value -> large similarity means small changes occurs from global model
        if self.training_settings['inverse']:
            score_vector = torch.exp(-similarity)
        else:
            score_vector = torch.exp(similarity)
        score_vector = torch.softmax(score_vector / self.training_settings['T'], dim=0)
        return score_vector

    def weight_apply(self, score_vector, layer_vector):
        size_differences = len(layer_vector.size()) - len(score_vector.size())
        for _ in range(size_differences):
            score_vector = score_vector.unsqueeze(-1)
        scored_vector = torch.sum(score_vector * layer_vector, dim=0) * self.training_settings['global_lr']
        return scored_vector


class FedDF(Aggregator):
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 **kwargs):

        super().__init__(test_data, valid_data, dataset_name, log_path, train_settings, **kwargs)
        self.test_accuracy = self.compute_accuracy()
        self.server_model: Module = model_call(train_settings['model'],
                                               NUMBER_OF_CLASSES[dataset_name],
                                               data_type=dataset_name.lower())

    def feddf(self, clients: List[Client], global_lr: float):
        # train and update
        self.server_model.train()
        empty_model = OrderedDict()

        total_len = sum(client.data_len() for client in clients)

        for k, v in self.model.state_dict().items():
            for client in clients:
                p = (client.data_len() / total_len)
                empty_model[k] = empty_model.get(k, 0) + client.model[k] * p * global_lr

        self.server_model.load_state_dict(empty_model)
        self.server_model = self.server_model.to(self.device)

        optimizer = torch.optim.Adam(self.server_model.parameters(), lr=1e-3)

        for _ in range(self.training_settings['server_epochs']):
            for _, (data, target) in enumerate(self.valid_loader):
                optimizer.zero_grad()
                # train model
                data, target = data.to(self.device), target.to(self.device)

                output = self.server_model(data)
                client_logits = []
                for client in clients:
                    self.client_model.load_state_dict(client.model)
                    _model = self.client_model.to(self.device)
                    client_logits.append(_model(data).detach())
                teacher_logits = sum(client_logits) / len(clients)

                loss = divergence(output, teacher_logits)
                loss.backward()
                optimizer.step()

        self.mark_model_diff(clients)

        # Global model updates
        self.set_parameters(self.server_model.state_dict())
        self.global_iter += 1


class FedBE(Aggregator):
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 **kwargs):

        super().__init__(test_data, valid_data, dataset_name, log_path, train_settings, **kwargs)
        self.test_accuracy = self.compute_accuracy()
        self.server_model: Module = model_call(train_settings['model'],
                                               NUMBER_OF_CLASSES[dataset_name],
                                               data_type=dataset_name.lower())
        self.swag_model: Module = model_call(train_settings['model'],
                                             NUMBER_OF_CLASSES[dataset_name],
                                             data_type=dataset_name.lower())

    def fedbe(self, clients: List[Client], global_lr: float):
        # generate teachers
        nets = []
        base_teachers = []
        prev_global_param = self.model.state_dict()

        empty_model = OrderedDict()
        total_len = sum(client.data_len() for client in clients)

        for k, v in self.model.state_dict().items():
            for client in clients:
                p = (client.data_len() / total_len)
                empty_model[k] = empty_model.get(k, 0) + client.model[k] * p * global_lr

        nets.append(empty_model)

        for client in clients:
            nets.append(client.model)
            base_teachers.append(client.model)

        # generate swag model
        swag_server = SWAG_server(prev_global_param, avg_model=empty_model, concentrate_num=1)
        w_swag = swag_server.construct_models(base_teachers, mode='gaussian')
        nets.append(w_swag)

        self.server_model.load_state_dict(empty_model)
        self.server_model = self.server_model.to(self.device)
        optimizer = torch.optim.SGD(self.server_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)

        logits = []
        # train and update
        self.server_model.train()
        # central_node.model.cuda().train()
        for _ in range(self.training_settings['server_epochs']):
            # train_loader = central_node.validate_set
            for _, (data, target) in enumerate(self.valid_loader):
                optimizer.zero_grad()
                # train model
                data, target = data.to(self.device), target.to(self.device)

                output = self.server_model(data)

                for weight in nets:
                    self.client_model.load_state_dict(weight)
                    self.client_model.to(self.device)
                    logits.append(self.client_model(data).detach())
                teacher_logits = sum(logits) / len(nets)

                loss = divergence(output, teacher_logits)
                loss.backward()
                optimizer.step()

        self.mark_model_diff(clients)
        # Global model updates
        self.set_parameters(self.server_model.state_dict())
        self.global_iter += 1


class SWAG_server(torch.nn.Module):
    def __init__(self, base_model, avg_model=None, max_num_models=25, var_clamp=1e-5, concentrate_num=1):
        self.base_model = base_model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.concentrate_num = concentrate_num
        self.avg_model = avg_model

    def compute_var(self, mean, sq_mean):
        var_dict = {}
        for k in mean.keys():
            var = torch.clamp(sq_mean[k] - mean[k] ** 2, self.var_clamp)
            var_dict[k] = var

        return var_dict

    def compute_mean_sq(self, teachers):
        w_avg = {}
        w_sq_avg = {}
        w_norm = {}

        for k in teachers[0].keys():
            if "batches_tracked" in k: continue
            w_avg[k] = torch.zeros(teachers[0][k].size())
            w_sq_avg[k] = torch.zeros(teachers[0][k].size())
            w_norm[k] = 0.0

        for k in w_avg.keys():
            if "batches_tracked" in k: continue
            for i in range(0, len(teachers)):
                grad = teachers[i][k].cpu() - self.base_model[k].cpu()
                norm = torch.norm(grad, p=2)

                grad = grad / norm
                sq_grad = grad ** 2

                w_avg[k] += grad
                w_sq_avg[k] += sq_grad
                w_norm[k] += norm

            w_avg[k] = torch.div(w_avg[k], len(teachers))
            w_sq_avg[k] = torch.div(w_sq_avg[k], len(teachers))
            w_norm[k] = torch.div(w_norm[k], len(teachers))

        return w_avg, w_sq_avg, w_norm

    def construct_models(self, teachers, mean=None, mode="dir"):
        if mode == "gaussian":
            w_avg, w_sq_avg, w_norm = self.compute_mean_sq(teachers)
            w_var = self.compute_var(w_avg, w_sq_avg)

            mean_grad = copy.deepcopy(w_avg)
            for i in range(self.concentrate_num):
                for k in w_avg.keys():
                    mean = w_avg[k]
                    var = torch.clamp(w_var[k], 1e-6)

                    eps = torch.randn_like(mean)
                    sample_grad = mean + torch.sqrt(var) * eps * 0.1
                    mean_grad[k] = (i * mean_grad[k] + sample_grad) / (i + 1)

            for k in w_avg.keys():
                mean_grad[k] = mean_grad[k] * 1.0 * w_norm[k] + self.base_model[k].cpu()

            return mean_grad

        elif mode == "random":
            num_t = 3
            ts = np.random.choice(teachers, num_t, replace=False)
            mean_grad = {}
            for k in ts[0].keys():
                mean_grad[k] = torch.zeros(ts[0][k].size())
                for i, t in enumerate(ts):
                    mean_grad[k] += t[k]

            for k in ts[0].keys():
                mean_grad[k] /= num_t

            return mean_grad

        elif mode == "dir":
            proportions = np.random.dirichlet(np.repeat(1.0, len(teachers)))
            mean_grad = {}
            for k in teachers[0].keys():
                mean_grad[k] = torch.zeros(teachers[0][k].size())
                for i, t in enumerate(teachers):
                    mean_grad[k] += t[k] * proportions[i]

            for k in teachers[0].keys():
                mean_grad[k] /= sum(proportions)

            return mean_grad

# class FedLAW(Aggregator):
#     def __init__(self,
#                  test_data: DataLoader,
#                  valid_data: DataLoader,
#                  dataset_name: str,
#                  log_path: str,
#                  train_settings: dict,
#                  **kwargs):
#
#         super().__init__(test_data, valid_data, dataset_name, log_path, train_settings, **kwargs)
#         self.test_accuracy = self.compute_accuracy()
#         self.server_model: Module = model_call(train_settings['model'],
#                                                NUMBER_OF_CLASSES[dataset_name],
#                                                data_type=dataset_name.lower())
#
#     def receive_client_models(self, client_nodes, select_list, size_weights):
#         client_params = []
#         for client in client_nodes:
#             client_params.append(client.model.get_param(clone=True))
#
#         agg_weights = [size_weights[idx] for idx in select_list]
#         agg_weights = [w / sum(agg_weights) for w in agg_weights]
#
#         return agg_weights, client_params
#
#     def fedlaw_optimization(self, clients: List[Client], global_lr: float):
#         '''
#         fedlaw optimization functions for optimize both gamma and lambdas
#         '''
#         cohort_size = len(clients)
#         # initialize gamma and lambdas
#         # the last element is gamma
#         empty_model = OrderedDict()
#
#         total_len = sum(client.data_len() for client in clients)
#
#         for k, v in self.model.state_dict().items():
#             for client in clients:
#                 p = (client.data_len() / total_len)
#                 empty_model[k] = empty_model.get(k, 0) + client.model[k] * p * global_lr
#
#         if self.training_settings['server_funct'] == 'exp':
#             optimizees = torch.tensor([torch.log(torch.tensor(j)) for j in empty_model.values()] + [0.0],
#                                       device=self.device,
#                                       requires_grad=True)
#         elif self.training_settings['server_funct'] == 'quad':
#             optimizees = torch.tensor([math.sqrt(1.0 / cohort_size) for j in empty_model.values()] + [1.0],
#                                       device=self.device,
#                                       requires_grad=True)
#         else:
#             raise KeyError('server_funct in training_settings is missing.')
#
#         optimizee_list = [optimizees]
#         optimizer = torch.optim.Adam(optimizee_list, lr=0.01, betas=(0.5, 0.999))
#
#         # set the scheduler
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#
#         # clear grad
#         for i in range(len(optimizee_list)):
#             optimizee_list[i].grad = torch.zeros_like(optimizee_list[i])
#
#         # Train optimizees
#         softmax = torch.nn.Softmax(dim=0)
#         # set the model as train to update the buffers for normalization layers
#         self.server_model.train().to(self.device)
#         for epoch in range(self.training_settings['server_epochs']):
#             # the training data is the small dataset on the server
#             for itr, (data, target) in enumerate(self.valid_loader):
#                 # Client model parameters
#                 for i in range(cohort_size):
#                     if i == 0:
#                         if self.training_settings['server_funct'] == 'exp':
#                             model_param = torch.exp(optimizees[-1]) * softmax(optimizees[:-1])[i] * parameters[i]
#                         elif self.training_settings['server_funct'] == 'quad':
#                             model_param = optimizees[-1] * optimizees[-1] * optimizees[i] * optimizees[i] / sum(
#                                 optimizees[:-1] * optimizees[:-1]) * parameters[i]
#                     else:
#                         if args.server_funct == 'exp':
#                             model_param = model_param.add(
#                                 torch.exp(optimizees[-1]) * softmax(optimizees[:-1])[i] * parameters[i])
#                         elif args.server_funct == 'quad':
#                             model_param = model_param.add(
#                                 optimizees[-1] * optimizees[-1] * optimizees[i] * optimizees[i] / sum(
#                                     optimizees[:-1] * optimizees[:-1]) * parameters[i])
#
#                 # train model
#                 data, target = data.cuda(), target.cuda()
#
#                 # Update optimizees
#                 # zero_grad
#                 optimizer.zero_grad()
#                 # update models according to the lr
#                 output = central_node.model.forward_with_param(data, model_param)
#                 loss = F.cross_entropy(output, target)
#                 loss.backward()
#                 optimizer.step()
#             # scheduling
#             scheduler.step()
#         # record and print current lam
#         if args.server_funct == 'exp':
#             optmized_weights = [j for j in softmax(optimizees[:-1]).detach().cpu().numpy()]
#             learned_gamma = torch.exp(optimizees[-1])
#         elif args.server_funct == 'quad':
#             optmized_weights = [j * j / sum(optimizees[:-1] * optimizees[:-1]) for j in
#                                 optimizees[:-1].detach().cpu().numpy()]
#             learned_gamma = optimizees[-1] * optimizees[-1]
#         return learned_gamma, optmized_weights