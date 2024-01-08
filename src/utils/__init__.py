from .data_loader import FedCifar, FedMNIST, FedOrganAMNIST, FedBloodMNIST


def dataset_call(name: str, log_path: str, **kwargs):
    if name.lower() == 'mnist':
        dataset_object = FedMNIST(log_path)
    elif 'cifar' in name.lower():
        dataset_object = FedCifar(log_path, mode=name.lower(), **kwargs)
    elif 'organamnist' in name.lower():
        dataset_object = FedOrganAMNIST(log_path, mode=name.lower(), **kwargs)
    elif 'bloodmnist' in name.lower():
        dataset_object = FedBloodMNIST(log_path, mode=name.lower(), **kwargs)
    else:
        return NotImplementedError
    return dataset_object.load(kwargs['num_of_clients'], dirichlet_alpha=kwargs['dirichlet_alpha'])
    # return fed_dataset, valid_loader, test_loader #fed_dataset, test_loader
