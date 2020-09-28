from typing import NamedTuple, List


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float
    pos_accuracy: float
    neg_accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    
    train_pos_acc: List[float]
    train_neg_acc: List[float]
    test_pos_acc: List[float]
    test_neg_acc: List[float]


class EpochHeatMap(NamedTuple):
    """
    Represents epoch data for generating Attention Map graph
    """
    y_vals: List[int]
    attention_map: List[float]
    indices_list: List[int]