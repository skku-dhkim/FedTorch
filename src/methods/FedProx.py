"""
This code is referred from 'https://github.com/JYWa/FedNova/tree/master'
"""

from . import *
from .utils import *
from src.methods.federated import run
from torch.optim.optimizer import Optimizer
from src.train.train_utils import client_gradient, compute_layer_norms


class FedProx(Optimizer):
    r"""Implements FedAvg and FedProx. Local Solver can have momentum.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
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

    def __init__(self, params, gmf, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0):

        self.gmf = gmf
        self.itr = 0
        self.a_sum = 0
        self.mu = mu

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
        super(FedProx, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedProx, self).__setstate__(state)
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

                # apply proximal update
                d_p.add_(p.data - param_state['old_init'], alpha=self.mu)
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int):
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, data_type=client.data_type)
    model.load_state_dict(client.model)
    model = model.to(device)

    original_state = F.get_parameters(model)

    # INFO - Optimizer
    optim = FedProx(
        model.parameters(),
        mu=0,
        gmf=0,
        lr=training_settings['local_lr'],
        momentum=training_settings['momentum'],
        nesterov=False,
        weight_decay=training_settings['weight_decay']
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        training_loss = 0
        summary_counter = 0

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device).to(torch.long)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)

            current_state = F.get_parameters(model)

            mu = training_settings['mu']
            proximal_term = 0.0

            for k in current_state.keys():
                proximal_term += (current_state[k] - original_state[k]).norm(2)

            loss = loss_fn(outputs, labels) + mu * proximal_term

            loss.backward()
            optim.step()

            # INFO - Step summary
            training_loss += loss.item()

            client.step_counter += 1
            summary_counter += 1

        # INFO - Epoch summary
        test_acc, test_loss = F.compute_accuracy(model, client.test_loader, loss_fn)
        train_acc, train_loss = F.compute_accuracy(model, client.train_loader, loss_fn)

        summary_writer.add_scalar('acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('acc/local_test', test_acc, client.epoch_counter)

        summary_writer.add_scalar('loss/local_train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('loss/local_test', test_loss, client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.grad = client_gradient(previous=client.model, current=model.state_dict())
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})

    # INFO - For the Aggregation Balancer
    if training_settings['aggregator'].lower() == 'balancer':
        client.grad_norm = compute_layer_norms(client.grad)

    return client


def run_fedprox(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
