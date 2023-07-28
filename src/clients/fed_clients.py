from src import *
from src.clients import *


class Client:
    def __init__(self,
                 client_name: str,
                 data: dict,
                 batch_size: int,
                 log_path: str):

        # Client Meta setting
        self.name = client_name

        # Data settings
        self.train = data['train']
        if len(self.train) < batch_size:
            self.train_loader = DataLoader(self.train,
                                           batch_size=batch_size, drop_last=False)
        else:
            self.train_loader = DataLoader(self.train,
                                           batch_size=batch_size, shuffle=True, drop_last=False)
        self.test = data['test']
        self.test_loader = DataLoader(self.test,
                                      batch_size=batch_size, shuffle=False, drop_last=False)

        # Training settings
        self.global_iter = []
        self.step_counter = 0
        self.epoch_counter = 0

        # Log path
        self.summary_path = os.path.join(log_path, "{}".format(client_name))

        # Model
        self.model: Optional[OrderedDict] = None

        # self.num_per_class = self.train.num_per_class

        # Cosine Similarity
        # self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)

    def data_len(self):
        return len(self.train)

# class FedKL_Client(Client):
#     def __init__(self,
#                  client_name: str,
#                  data: dict,
#                  batch_size: int,
#                  log_path: str):
#         super().__init__(client_name, data, batch_size, log_path)
#         self.F_entropy_gap = None
#         self.O_entropy_gap = None


class FedBalancerClient(Client):
    def __init__(self,
                 client_name: str,
                 data: dict,
                 batch_size: int,
                 log_path: str):
        super().__init__(client_name, data, batch_size, log_path)
        self.num_per_class = self.train.num_per_class

    # def activation_counter_clear(self):
    #     self.activation_counts = None
