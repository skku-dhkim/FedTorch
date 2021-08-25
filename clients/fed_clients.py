from dataset.data_loader import DatasetWrapper


class Client:
    def __init__(self, client_name: str, train_data, test_data=None):
        self.name = client_name
        self.train = train_data
        self.dataset = DatasetWrapper(data=train_data)
        self.train_loader = None
        self.model = None
        self.training_loss = 0.0
        self.global_iter = 0

        if test_data:
            self.test_dataset = test_data

