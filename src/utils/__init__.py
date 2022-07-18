from .data_loader import FedCifar, FedMNIST


def dataset_call(name: str, log_path: str, **kwargs):
    if name.lower() == 'mnist':
        dataset_object = FedMNIST(log_path)
        fed_dataset, valid_loader, test_loader = dataset_object.load(kwargs['num_of_clients'],
                                                       dirichlet_alpha=kwargs['dirichlet_alpha'])
    elif name.lower() == 'cifar-10':
        dataset_object = FedCifar(log_path, mode=name.lower(), **kwargs)
        fed_dataset, valid_loader, test_loader = dataset_object.load(kwargs['num_of_clients'],
                                                       dirichlet_alpha=kwargs['dirichlet_alpha'])
    elif name.lower() == 'cifar-100':
        dataset_object = FedCifar(log_path, mode=name.lower(), **kwargs)
        fed_dataset, valid_loader, test_loader = dataset_object.load(kwargs['num_of_clients'],
                                                       dirichlet_alpha=kwargs['dirichlet_alpha'])
    else:
        return NotImplementedError

    return fed_dataset, valid_loader, test_loader #fed_dataset, test_loader
