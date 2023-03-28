from src.train import *


def get_probability(input_tensor: torch.Tensor, logit: bool = False) -> torch.Tensor:
    """
    Make probability of input tensor
    Args:
        input_tensor: (torch.Tensor) Tensor to make probability density.
        logit: (boolean) To check logit or not.

    Returns: (torch.Tensor) Probability density tensor.
    """
    if logit:
        input_tensor = torch.flatten(input_tensor, 1)
        output = nn.functional.softmax(input_tensor, dim=-1)
    else:
        input_tensor = torch.flatten(input_tensor, 2)
        output = input_tensor / (torch.sum(input_tensor, dim=-1, keepdim=True)+1e-7)
    return output


def min_max_normalization(input_tensor: torch.Tensor,
                          T: float = 1.0,
                          alpha: float = 0.1, beta: float = 1.0,
                          minimum: Union[Union[float, torch.Tensor], None] = None,
                          maximum: Union[Union[float, torch.Tensor], None] = None) -> torch.Tensor:
    """
    Min-Max normalization function. Alpha and beta used for the range of normalized input.
    Args:
        input_tensor: (torch.Tensor) Input tensor.
        T: (float) temperature value of normalization
        alpha: (float) Minimum range of normalized input.
        beta: (float) Maximum range of normalized input.
        minimum: (Union[float, torch.Tensor]): Minimum range.
        maximum: (Union[float, torch.Tensor]): Maximum range.
    Returns: (torch.Tensor) Min-Max normalized tensor.

    """
    if minimum is None:
        minimum = torch.min(input_tensor, dim=-1, keepdim=True)[0]
    if maximum is None:
        maximum = torch.max(input_tensor, dim=-1, keepdim=True)[0]

    output = (input_tensor - minimum) / (((maximum - minimum)*(beta-alpha) + alpha) / T)
    return output


def entropy(input_tensor: torch.Tensor, base: Union[int, str] = 2, normalize: bool = False) -> torch.Tensor:
    """

    Args:
        input_tensor: (torch.Tensor) Input tensor; should be form as probability tensor.
        base: (Union[int, str]) Log type. Either base 2 log or ln.
        normalize: (boolean) Returns normalized entropy or not.

    Returns: (torch.Tensor) Entropy of input tensor that normalized or original .

    """
    if base == 2:
        output = -torch.sum(input_tensor*torch.nan_to_num(torch.log2(input_tensor+1e-7)), dim=-1)
    else:

        output = -torch.sum(input_tensor*torch.nan_to_num(torch.log(input_tensor+1e-7)), dim=-1)
    if normalize:
        max_entr = max_entropy(input_tensor, base=base)
        output = min_max_normalization(output, alpha=0, minimum=0, maximum=max_entr)
    return output


def max_entropy(input_tensor: torch.Tensor, base: Union[int, str]) -> torch.Tensor:
    """
    Calculate maximum entropy of input tensor.
    This function is to have possible maximum entropy of input tensor.
    Not to have the biggest entropy value of input.
    Args:
        input_tensor: (torch.Tensor) Probability tensor of input
        base: (Union[int, str]) Log type. Either base 2 log or ln.

    Returns: (torch.Tensor) This calculates the maximum entropy possible,
                            which means that all probability distributions have equal values.
    """
    num_classes = input_tensor.shape[-1]
    p = torch.full((1, num_classes), 1/num_classes)
    if base == 2:
        log_max_entropy = -torch.sum(p * torch.log2(p))
    else:
        log_max_entropy = -torch.sum(p * torch.log(p))
    return log_max_entropy


def vector_normalization(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Vector normalization method.
    Args:
        input_tensor: (torch.Tensor) Input tensor

    Returns: (torch.Tensor) Vector normalized tensor.
    """
    output = input_tensor / torch.sqrt(torch.sum(torch.sqrt(input_tensor), dim=-1, keepdim=True))
    return output


def sum_mean(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce dimension with sum after mean operation.
    Args:
        input_tensor: (torch.Tensor) Input tensor.

    Returns: (torch.Tensor) Reduced tensor.

    """
    output = torch.mean(torch.sum(input_tensor, dim=-1), dim=-1)
    return output


def mean_mean(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce dimension with mean after mean operation.
    Args:
        input_tensor: (torch.Tensor) Input tensor.

    Returns: (torch.Tensor) Reduced tensor.

    """
    output = torch.mean(torch.mean(input_tensor, dim=-1), dim=-1)
    return output


def l1_norm(input_tensor: torch.Tensor, logit: bool = False) -> torch.Tensor:
    """
    Calculate the L1 norm from input tensor
    Args:
        input_tensor: (torch.Tensor) Input tensors to calculate the l1-norm
        logit: (boolean) Check either logit or not.

    Returns: (torch.Tensor) output of input norm

    """
    if logit:
        input_tensor = torch.flatten(input_tensor, 1)
        output = input_tensor.norm(p=1, dim=1)
    else:
        input_tensor = torch.flatten(input_tensor, 2)
        output = input_tensor.norm(p=1, dim=2)
    return output


def l2_norm(input_tensor: torch.Tensor, logit: bool = False):
    """
    Calculate the L2 norm from input tensor
    Args:
        input_tensor: (torch.Tensor) Input tensors to calculate the l1-norm
        logit: (boolean) Check either logit or not.

    Returns: (torch.Tensor) output of input norm
    """
    if logit:
        input_tensor = torch.flatten(input_tensor, 1)
        output = input_tensor.norm(p=2, dim=1)
    else:
        input_tensor = torch.flatten(input_tensor, 2)
        output = input_tensor.norm(p=2, dim=2)
    return output


def loss_fn_kd(input_distribution: torch.Tensor, target_distribution: torch.Tensor,
               alpha: Union[torch.Tensor, float] = 1.0, temperature: float = 2.0) -> torch.Tensor:
    """
    Knowledge distillation function. Use the KL divergence loss with input and target distribution.
    Args:
        input_distribution: (torch.Tensor) Input distribution tensor.
        target_distribution: (torch.Tensor) Target distribution tensor.
        alpha: (Union[torch.Tensor, float]) Balance parameter of KD loss.
        temperature: (float) Balance parameter of soft target smoothness.

    Returns: (torch.Tensor) KD loss of two distribution.
    """
    log_softmax_input = nn.functional.log_softmax(input_distribution / temperature, dim=1)
    log_softmax_target = nn.functional.log_softmax(target_distribution / temperature, dim=1)
    KD_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
    KD_loss = KD_loss_fn(log_softmax_input, log_softmax_target) * (alpha * temperature * temperature)
    return torch.nan_to_num(KD_loss)


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Do one-hot encoding for labels.
    Args:
        labels: (torch.Tensor) Label tensor.
        num_classes: (int) number of classes of data.
    Returns: (torch.Tensor) one_hot encoded tensor.
    """
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

