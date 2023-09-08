"""
This code is referred from 'https://github.com/JYWa/FedNova/tree/master'
"""
import torch.nn

from src.methods import *
from .utils import *
from torch.optim.optimizer import Optimizer
from .FedBalancer import aggregation_balancer
from src.clients import AggregationBalancer


class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params,  gmf, mu=0, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):

        self.gmf = gmf
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(p.data - param_state['old_init'], alpha=self.mu)

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(d_p, alpha=local_lr)

                p.data.add_(d_p, alpha=-local_lr)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu

        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss


@ray.remote(max_calls=1)
def train(client: Client, training_settings: dict, num_of_classes: int):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))
    csvfile = open(os.path.join(client.summary_path, "experiment_result.csv"), "a", newline='')
    csv_writer = csv.writer(csvfile)

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, features=False)
    model.load_state_dict(client.model)
    model = model.to(device)

    # INFO - Optimizer
    optim = FedNova(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_settings['local_lr'],
        gmf=0,
        mu=0,
        momentum=training_settings['momentum'],
        nesterov=False,
        weight_decay=training_settings['weight_decay']
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # INFO: Local training logic
    for i in range(training_settings['local_epochs']):

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optim.step()

        client.epoch_counter += 1
        training_acc, training_losses = F.compute_accuracy(model, client.train_loader, loss_fn)
        test_acc, test_losses = F.compute_accuracy(model, client.test_loader, loss_fn)

        # INFO - Epoch summary
        summary_writer.add_scalar('acc/train', training_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/train', training_losses, client.epoch_counter)
        summary_writer.add_scalar('acc/test', test_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/test', test_losses, client.epoch_counter)

        csv_writer.writerow([training_acc, training_losses, test_acc, test_losses])

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    csvfile.close()
    return client


def local_training(clients: list,
                   training_settings: dict,
                   num_of_class: int) -> list:
    """
    Args:
        clients: (list) client list to join the training.
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
    Returns: (List) Client Object result

    """
    ray_jobs = []
    for client in clients:
        if training_settings['use_gpu']:
            ray_jobs.append(train.options(num_gpus=training_settings['gpu_frac']).remote(client,
                                                                                         training_settings,
                                                                                         num_of_class))
        else:
            ray_jobs.append(train.options().remote(client,
                                                   training_settings,
                                                   num_of_class))
    trained_result = []
    while len(ray_jobs):
        done_id, ray_jobs = ray.wait(ray_jobs)
        trained_result.append(ray.get(done_id[0]))

    return trained_result


def fed_avg(clients: List[Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    total_len = 0
    empty_model = OrderedDict()

    for client in clients:
        total_len += client.data_len()

    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            if k not in empty_model.keys():
                empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr
            else:
                empty_model[k] += client.model[k] * (client.data_len() / total_len) * global_lr

    # Global model updates
    aggregator.set_parameters(empty_model)
    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()

    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def run(client_setting: dict, training_setting: dict):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = Client

    if training_setting['balancer'] is True:
        aggregator: type(AggregationBalancer) = AggregationBalancer
    else:
        aggregator = Aggregator

    clients, aggregator = client_initialize(client, aggregator, fed_dataset, test_loader, valid_loader,
                                            client_setting, training_setting)
    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[4] Global step starts...")

        pbar = tqdm(range(training_setting['global_epochs']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        lr_decay = False
        if 'lr_decay' in training_setting.keys():
            initial_lr = training_setting['local_lr']
            total_g_epochs = training_setting['global_epochs']
            lr_decay = True

        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - Save the global model
            aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            stream_logger.debug("[*] Client sampling...")
            # INFO - Client sampling
            sampled_clients = F.client_sampling(clients, sample_ratio=training_setting['sample_ratio'], global_round=gr)

            # INFO - Learning rate decay
            if lr_decay:
                if 'cos' in training_setting['lr_decay'].lower():
                    # INFO - COS decay
                    training_setting['local_lr'] = 1 / 2 * initial_lr * (
                                1 + math.cos(aggregator.global_iter * math.pi / total_g_epochs))
                    training_setting['local_lr'] = 0.001 if training_setting['local_lr'] < 0.001 else training_setting['local_lr']
                elif 'manual' in training_setting['lr_decay'].lower():
                    if aggregator.global_iter in [total_g_epochs // 4, (total_g_epochs * 3) // 8]:
                        training_setting['local_lr'] *= 0.1
                else:
                    raise NotImplementedError("Learning rate decay \'{}\' is not implemented yet.".format(
                        training_setting['lr_decay']))

            stream_logger.debug("[*] Learning rate decay: {}".format(training_setting['local_lr']))
            summary_logger.info("[{}/{}] Current local learning rate: {}".format(aggregator.global_iter,
                                                                                 total_g_epochs,
                                                                                 training_setting['local_lr']))

            # INFO - Local Training
            stream_logger.debug("[*] Local training process...")
            trained_clients = local_training(clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])

            stream_logger.debug("[*] Federated aggregation scheme...")
            if training_setting['balancer'] is True:
                stream_logger.debug("[*] Aggregation Balancer")
                aggregation_balancer(trained_clients, aggregator, training_setting['global_lr'], training_setting['T'])
            else:
                stream_logger.debug("[*] FedAvg")
                fed_avg(trained_clients, aggregator, training_setting['global_lr'])

            stream_logger.debug("[*] Weight Updates")

            clients = F.update_client_dict(clients, trained_clients)

            end_time_global_iter = time.time()
            pbar.set_postfix({'global_acc': aggregator.test_accuracy})
            summary_logger.info("Global Running time: {}::{:.2f}".format(gr,
                                                                         end_time_global_iter - start_time_global_iter))
            summary_logger.info("Test Accuracy: {}".format(aggregator.test_accuracy))
        summary_logger.info("Global iteration finished successfully.")
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    end_run_time = time.time()
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))
    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")