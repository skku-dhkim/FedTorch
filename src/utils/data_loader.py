from .. import *
from .logger import write_experiment_summary
from torchvision.datasets import *
from torchvision.transforms import *
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

import copy
import seaborn as sns


class CustomDataLoader:
    def __init__(self, train_data, test_data, log_path, transform=None):
        self.train_X = train_data.data
        self.train_Y = np.array(train_data.targets)

        self.test_X = test_data.data
        self.test_Y = np.array(test_data.targets)

        self.valid_set = {'x': [], 'y': []}

        self.num_of_categories = len(train_data.classes)
        self.categories_train_X, self.categories_train_Y = None, None

        self.main_dir = Path(log_path).parent.absolute()
        self.log_path = os.path.join(log_path, "client_meta")
        os.makedirs(self.log_path, exist_ok=True)
        self.transform = transform

    def _data_sampling(self, dirichlet_alpha: float, num_of_clients: int, num_of_classes: int) -> pd.DataFrame:
        """
        Args:
            dirichlet_alpha: (float) adjusting iid-ness. higher the alpha, more iid-ness distribution.
            num_of_clients: (int) number of clients
            num_of_classes: (int) number of classes
        :return:
            DataFrame: Client data distribution for iid-ness.
        """
        # Get dirichlet distribution
        # NOTE: If doesn't have sub-directory, manual seed is used.
        if self.main_dir == Path("./logs").absolute():
            seed = 2023
        elif os.path.isfile(os.path.join(self.main_dir, 'seed.txt')):
            with open(os.path.join(self.main_dir, 'seed.txt'), 'r') as f:
                seed = f.read().strip()
                seed = int(seed)
        else:
            with open(os.path.join(self.main_dir, 'seed.txt'), 'w') as f:
                seed = random.randint(1, int(1e+4))
                f.write(str(seed))
        write_experiment_summary("Data sampling", {"Seed": seed})

        np.random.seed(seed)
        s = np.random.dirichlet(np.repeat(dirichlet_alpha, num_of_clients), num_of_classes)
        c_dist = pd.DataFrame(s)

        # Round for data division convenience.
        c_dist = c_dist.round(2)
        while len(c_dist.columns[(c_dist == 0).all()]) > 0:
            s = np.random.dirichlet(np.repeat(dirichlet_alpha, num_of_clients), num_of_classes)
            c_dist = pd.DataFrame(s)
            # Round for data division convenience.
            c_dist = c_dist.round(2)

        sns.set(rc={'figure.figsize': (20, 20)})
        ax = sns.heatmap(c_dist, cmap='YlGnBu', annot=False)
        ax.set(xlabel='Clients', ylabel='Classes')
        figure = ax.get_figure()

        # Save to Image
        figure.savefig(os.path.join(self.log_path, 'client_meta.png'), format='png')
        c_dist.to_csv(os.path.join(self.log_path, 'client_meta.csv'), index=False)

        # Tensorboard log
        self.summary_writer = SummaryWriter(self.log_path)
        self.summary_writer.add_figure("client_meta", figure)

        return c_dist.transpose()

    def _data_proportion_allocate(self, clients: list, proportion: pd.DataFrame) -> list:
        """
        :param clients: (list) Client lists
        :param proportion: (DataFrame) Data proportion for every client on every labels.
        :return:
            list: Train dataset for every client.
        """
        # Initialize index manager. This is for start and last index managing.
        idx_manage = {}
        for i in range(proportion.shape[1]):
            idx_manage[i] = 0

        # Start allocating data
        for idx, client in enumerate(clients):
            distribution = proportion.iloc[idx]
            for k, dist in enumerate(distribution):
                num_of_data = int(len(self.categories_train_X[k]) * dist)
                client['train']['x'].append(self.categories_train_X[k][idx_manage[k]:idx_manage[k] + num_of_data])
                client['train']['y'].append(self.categories_train_Y[k][idx_manage[k]:idx_manage[k] + num_of_data])
                # Update Last index number. It will be first index at next iteration.
                idx_manage[k] = idx_manage[k] + num_of_data

            # Make an integrated array.
            client['train']['x'] = np.concatenate(client['train']['x'])
            client['train']['y'] = np.concatenate(client['train']['y'])

            # Make random index list
            index = [j for j in range(len(client['train']['x']))]
            random.shuffle(index)

            client['train']['x'] = client['train']['x'][index]
            client['train']['y'] = client['train']['y'][index]

        return clients

    def _categorize(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        :param x: (numpy.ndarray) input x
        :param y: (numpy.ndarray) label y
        :return:
            tuple: (categories_X: dict, categories_Y: dict)
        """
        categories_X: Dict[int, Optional[Any]] = {}
        categories_Y: Dict[int, Optional[Any]] = {}

        for i in range(self.num_of_categories):
            # Get category index
            category_index = np.where(y == i)[0]
            categories_X[i] = x[category_index]
            categories_Y[i] = y[category_index]

            # Make random index list
            index = [j for j in range(len(categories_X[i]))]
            random.shuffle(index)

            # Apply random shuffling with same index number
            categories_X[i] = categories_X[i][index]
            categories_Y[i] = categories_Y[i][index]

        return categories_X, categories_Y

    def _to_dataset(self, clients: list, validation_split: float = 0.1) -> list:
        for client in clients:
            indices = int(len(client['train']['x']) * validation_split)

            train_x = client['train']['x'][:-indices]
            train_y = client['train']['y'][:-indices]

            valid_x = client['train']['x'][-indices:]
            valid_y = client['train']['y'][-indices:]

            client['train'] = DatasetWrapper({'x': train_x, 'y': train_y},
                                             transform=self.transform,
                                             number_of_categories=self.num_of_categories)
            for idx in range(len(valid_x)):
                self.valid_set['x'].append(valid_x[idx])
                self.valid_set['y'].append(valid_y[idx])

            client['test'] = DatasetWrapper({'x': valid_x, 'y': valid_y},
                                            transform=self.transform,
                                            number_of_categories=self.num_of_categories)
        return clients

    def load(self, number_of_clients: int, dirichlet_alpha: float) -> tuple:
        """
        Args
            number_of_clients: (int) Number of client to join federated learning.
            dirichlet_alpha: (float) Dirichlet distribution alpha. Greater the value more iid-ness data distribution.
        :return:
            tuple: (list: Client data set with non-iid setting, DataLoader: Test set loader)
        """
        # 1. Client definition and matching classes and collect validation set
        clients = [{'train': {'x': [], 'y': []}, 'valid': {'x': [], 'y': []}} for _ in range(number_of_clients)]

        # 2. Categorization of dataset
        self.categories_train_X, self.categories_train_Y = self._categorize(self.train_X, self.train_Y)

        # 3. Get data separation distribution
        client_distribution = self._data_sampling(dirichlet_alpha=dirichlet_alpha,
                                                  num_of_clients=number_of_clients,
                                                  num_of_classes=self.num_of_categories)

        # 4. Data allocation
        federated_dataset = self._data_proportion_allocate(clients, proportion=client_distribution)
        federated_dataset = self._to_dataset(federated_dataset)
        valid_loader = DataLoader(DatasetWrapper(self.valid_set, transform=self.transform), batch_size=16)
        # INFO - IID dataset
        test_loader = DataLoader(DatasetWrapper({'x': self.test_X, 'y': self.test_Y},
                                                transform=self.transform), batch_size=16)

        return federated_dataset, valid_loader, test_loader


class FedMNIST(CustomDataLoader):
    def __init__(self, log_path):
        train_data = MNIST(
            root="./data",
            train=True,
            download=True
        )
        test_data = MNIST(
            root="./data",
            train=False,
            download=True
        )
        CustomDataLoader.__init__(self, train_data, test_data, log_path, Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ]))


class FedCifar(CustomDataLoader):
    def __init__(self, log_path, **kwargs):
        if kwargs['mode'.lower()] == 'cifar-10':
            normalize = Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            train_data = CIFAR10(
                root="./data",
                train=True,
                download=True
            )
            test_data = CIFAR10(
                root="./data",
                train=False,
                download=True,
            )
            CustomDataLoader.__init__(self, train_data, test_data, log_path, Compose([
                ToTensor(),
                normalize
            ]))

        elif kwargs['mode'.lower()] == 'cifar-100':
            normalize = Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            train_data = CIFAR100(
                root="./data",
                train=True,
                download=True
            )
            test_data = CIFAR100(
                root="./data",
                train=False,
                download=True
            )
            CustomDataLoader.__init__(self, train_data, test_data, log_path, Compose([
                ToTensor(),
                normalize
            ]))
        else:
            raise ValueError("Invalid Parameter \'{}\'".format(kwargs['mode'.lower()]))


class DatasetWrapper(Dataset):
    def __init__(self, data, transform=None, number_of_categories=10):
        self.data_x = data['x']
        self.data_y = data['y']
        self.transform = transform
        self.num_per_class = [0] * number_of_categories
        self.class_list()

    def __len__(self) -> int:
        return len(self.data_x)

    def __getitem__(self, item) -> tuple:
        x, y = self.data_x[item], self.data_y[item]
        if self.transform:
            copied_x = copy.deepcopy(x)
            x = self.transform(copied_x)
        return x, y

    def class_list(self):
        _class_count = Counter(self.data_y)
        for k, v in _class_count.items():
            self.num_per_class[int(k)] = v
