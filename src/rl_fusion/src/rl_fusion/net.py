import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

from tianshou.data import to_torch

class Net(nn.Module):
    """Simple MLP backbone.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN), defaults to False.
    :param norm_layer: use which normalization before ReLU, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to None.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: tuple,
        action_shape: Optional[Union[tuple, int]] = 0,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        hidden_layer_size: int = 128,
        dueling: Optional[Tuple[int, int]] = None,
        norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dueling = dueling
        self.softmax = softmax
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)

        model = miniblock(input_size, hidden_layer_size, norm_layer)

        for i in range(layer_num):
            model += miniblock(
                hidden_layer_size, hidden_layer_size, norm_layer)

        if dueling is None:
            if action_shape and not concat:
                model += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
        else:  # dueling DQN
            q_layer_num, v_layer_num = dueling
            Q, V = [], []

            for i in range(q_layer_num):
                Q += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)
            for i in range(v_layer_num):
                V += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)

            if action_shape and not concat:
                Q += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
                V += [nn.Linear(hidden_layer_size, 1)]

            self.Q = nn.Sequential(*Q)
            self.V = nn.Sequential(*V)
        self.model = nn.Sequential(*model)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten -> logits."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), -1)
        logits = self.model(s)
        if self.dueling is not None:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            logits = q - q.mean(dim=1, keepdim=True) + v
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state