from random import sample
from typing import Tuple, Any

import torch


class ReplayBuffer:
    def __init__(self):
        self.size = 10**6
        self._buffer = []
        self.position = 0
    # migt have to change type of replay buffer
    def get_buffer(self) -> list:
        return self._buffer
    def clear_buffer(self) -> None:
        self._buffer = []
    def __len__(self):
        return len(self.get_buffer())

    def add(self, state, action, reward, next_state) -> None:
        if len(self) == self.size:
            # self._buffer[self.position] = torch.Tensor([state, action, reward, next_state])
            self._buffer[self.position] = [state, action, reward, next_state]
            self.position += 1
            self.position %= self.size
        else:
            # TODO: how to add to tensor
            # maybe dont try tensor, it will over complicate things
            self._buffer.append([state, action, reward, next_state])
            # self._buffer = torch.cat(torch.Tensor([state, action, reward, next_state]),
            #                          self._buffer, dim=0)

    # number
    # of
    # samples
    # per
    # minibatch
    # seems to be working, debugged it
    def sample(self, batch_size:int) -> tuple[Any, Any, Any, Any]:
        sampled_batch = sample(self._buffer, batch_size)
        states, actions, rewards, next_states = zip(*sampled_batch)
        return states, actions, rewards, next_states


