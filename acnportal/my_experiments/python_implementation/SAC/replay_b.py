from random import sample
from typing import Tuple, Any

import torch
from torch import Tensor


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

    def add(self, state, action, reward, next_state, done) -> None:

        # maybe convert them to other types, but this is used for basic compatibility
        # reward = [reward]
        # done = [done]
        # state = list(state)
        # next_state = list(next_state)
        # done = list(done)
        if len(self) == self.size:
            # self._buffer[self.position] = torch.Tensor([state, action, reward, next_state])
            self._buffer[self.position] = [state, action, reward, next_state, done]
            self.position += 1
            self.position %= self.size
        else:
            # TODO: how to add to tensor
            # maybe dont try tensor, it will over complicate things
            self._buffer.append([state, action, reward, next_state, done])
            # self._buffer = torch.cat(torch.Tensor([state, action, reward, next_state]),
            #                          self._buffer, dim=0)

    # number
    # of
    # samples
    # per
    # minibatch
    # seems to be working, debugged it
    def sample(self, batch_size:int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sampled_batch = sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sampled_batch)
        # convert everything not to tensor

        return (torch.Tensor(states),
                torch.stack(actions),
                torch.Tensor(rewards),
                torch.Tensor(next_states),
                torch.Tensor(dones))


