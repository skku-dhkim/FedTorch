import torch

from src import *
from src.clients import *
from src.model import *


class Aggregator:
    def __init__(self,
                 test_data: DataLoader,
                 valid_data: DataLoader,
                 dataset_name: str,
                 log_path: str,
                 train_settings: dict,
                 num_of_clients: int,
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
        self.model: FederatedModel = model_call(train_settings['model'], NUMBER_OF_CLASSES[dataset_name])

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
        self.summary_writer.add_scalar('global_test_acc', self.test_accuracy, self.global_iter)
        self.original_rep = self.compute_representations()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)

        self.num_of_clients = num_of_clients
        self.lrs = [0.1 for i in range(num_of_clients)]
        self.sim_l = [1.0 for i in range(num_of_clients)]
    @property
    def collected_weights(self):
        return self.__collected_weights

    @collected_weights.setter
    def collected_weights(self, value):
        self.__collected_weights = value

    def set_parameters(self, state_dict: Union[OrderedDict, dict]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ordict: bool = True) -> Union[OrderedDict, list]:
        if ordict:
            return OrderedDict({k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()})
        else:
            return [val.clone().detach().cpu() for _, val in self.model.state_dict().items()]
    
    def get_lrs(self):
        return self.lrs
    def get_sim(self):
        return self.sim_l

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
        layer_names = [name for name, module in self.model.named_children()]
        if 'fc' in layer_names:
            index = layer_names.index('fc')
        elif 'classifier' in layer_names:
            index = layer_names.index('classifier')
        else:
            raise ValueError("Unsupported layer name. Either \'fc\' or \'classifier\' supports.")

        features = layer_names[index-1]
        for name, module in self.model.named_children():
            if name == features:
                module.register_forward_hook(get_representations('rep'))

        self.model.eval()
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                _ = self.model(x)
                rep = torch.cat([rep, representations['rep'].reshape((x.shape[0], -1))], dim=0)
        return rep

    def fedAvg(self, model_save: bool = False):
        total_len = 0
        empty_model = OrderedDict()
        original_model = self.get_parameters()

        for _, client in self.collected_weights.items():
            total_len += client['data_len']
        
        self.lrs = self.fixed_lr(original_model)
        self.sim_l=self.sim_list(original_model)

        for k, v in self.model.state_dict().items():
            for _, client in self.collected_weights.items():
                if k not in empty_model.keys():
                    empty_model[k] = original_model[k] + (client['weights'][k] - original_model[k]) *self.lr*self.lrs[int(_)]
                else:
                    empty_model[k] += (client['weights'][k] - original_model[k]) *self.lr*self.lrs[int(_)]

        # Global model updates
        self.set_parameters(empty_model)
        self.global_iter += 1

        self.test_accuracy = self.compute_accuracy()

        # Calculate cos_similarity with previous representations
        self.calc_rep_similarity()

        # Calculate cos_similarity of weights
        current_model = self.get_parameters()
        self.calc_cos_similarity(original_model, current_model)
        self.summary_writer.add_scalar('global_test_acc', self.test_accuracy, self.global_iter)

        if model_save:
            self.save_model()

    ##proposed method
    ## method for learning rate adaptation based on fixed cosine value
    def fixed_lr(self, original_state: OrderedDict,target_sim = 0.99):#99999
        original_params = [] ##glob_w
        local_params = [[] for _, client in self.collected_weights.items()] ##
        grad_params = [[] for _, client in self.collected_weights.items()] ##calc_grad
        
        lr_list = [[] for _, client in self.collected_weights.items()]
        
        for k, v in self.model.state_dict().items():
            original_params.append(torch.flatten(original_state[k].to(torch.float32)))
            for _, client in self.collected_weights.items():
                local_params[int(_)].append(torch.flatten(client['weights'][k].to(torch.float32)))
            
        original_params = torch.cat(original_params) ## flatten and change to vector 
        
        for _,client in self.collected_weights.items():
            local_params[int(_)]= torch.cat(local_params[int(_)]) ## flatten and change to vector
            grad_params[int(_)] = local_params[int(_)] - original_params
        
        ## calculate learning rate on each clients
        summ = 0.0
        
        for _,client in self.collected_weights.items():
            A = torch.dot(grad_params[int(_)],original_params)
            G = torch.dot(original_params,original_params)
            dG = torch.dot(grad_params[int(_)], grad_params[int(_)])
            T = target_sim**2
            C = self.cos_sim(grad_params[int(_)],original_params)**2
            
            dtr = ((1.0-T)*A)**2 + G*dG*(1.0-T)*T*(T-C)
            sq = torch.sqrt(dtr)
            head = (1.0-T)*A +sq
            body = (T-C) * dG ##warning!! this part coud be "too small" to be denominator
            
            
            
            X = head/body
            if torch.isnan(X) or torch.isinf(X):
                X = 10.0

            #result = head/body ##this part should be clipped
            result = torch.clamp(X,-10.0,10.0)
            lr_list[int(_)] = result

        
        return lr_list 


    def calc_cos_similarity(self, original_state: OrderedDict, current_state: OrderedDict) -> Optional[OrderedDict]:
        result = OrderedDict()
        for k in current_state.keys():
            score = self.cos_sim(torch.flatten(original_state[k].to(torch.float32)),
                                 torch.flatten(current_state[k].to(torch.float32)))
            result[k] = score
            self.summary_writer.add_scalar("weight_similarity_before/{}".format(k), score, self.global_iter)
            self.summary_writer.add_scalar("weight_similarity_after/{}".format(k), score, self.global_iter)
        return result

    def calc_rep_similarity(self) -> torch.Tensor:
        current_rep = self.compute_representations()
        rd = self.cos_sim(self.original_rep, current_rep).cpu()
        score = torch.mean(rd)
        self.summary_writer.add_scalar("rep_similarity_before", score, self.global_iter)
        self.summary_writer.add_scalar("rep_similarity_after", score, self.global_iter)
        self.original_rep = current_rep
        return score

    def sim_list(self, original_state: OrderedDict,target_sim = 1 - (1e-15)):#99999
        original_params = [] ##glob_w
        local_params = [[] for _, client in self.collected_weights.items()] ##
        
        sim_list = [[] for _, client in self.collected_weights.items()]
        
        for k, v in self.model.state_dict().items():
            original_params.append(torch.flatten(original_state[k].to(torch.float32)))
            for _, client in self.collected_weights.items():
                local_params[int(_)].append(torch.flatten(client['weights'][k].to(torch.float32)))
            
        original_params = torch.cat(original_params) ## flatten and change to vector 
        
        for _,client in self.collected_weights.items():
            local_params[int(_)]= torch.cat(local_params[int(_)]) ## flatten and change to vector
               
        for _,client in self.collected_weights.items():
            C = self.cos_sim(local_params[int(_)],original_params)
            sim_list[int(_)] = C
        
        return sim_list
