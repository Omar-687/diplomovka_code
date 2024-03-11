import numpy as np
import torch
from torch import nn
from torch import distributions as dist
class PolicyNetwork(nn.Module):
    # he policy as a Gaussian with mean
    # and covariance given by neural networks
    # gaussian

    # multi-modal gaussian - need to check more in papers

    # univariate alebo multivariate?
    # it seems it is multivariate gaussian from videos, check them more precisely
    # All normal distributions can be described by just two parameters: the mean and the standard deviation.

    def __init__(self, state_dim, action_dim, neurons_per_hidden_layer=256):
        super(PolicyNetwork, self).__init__()
        self.activation_function = nn.ReLU()
        self.input_layer = nn.Linear(in_features=state_dim, out_features=neurons_per_hidden_layer)
        self.hidden_layer_1 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.hidden_layer_2 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.output_layer = nn.Linear(in_features=neurons_per_hidden_layer, out_features=action_dim)

        # mean and covariance for every element of action is needed
        self.mean_layer = nn.Linear(in_features=neurons_per_hidden_layer, out_features=action_dim)
        self.covariance_layer = nn.Linear(in_features=neurons_per_hidden_layer, out_features=action_dim)
        self.initialize_weights()
        self.initialize_biases()



    def initialize_weights(self) -> None:
        # nn.init.normal_(self.input_layer.weight, mean=0, std=0.01)
        # nn.init.normal_(self.hidden_layer_1.weight, mean=0, std=0.01)
        # nn.init.normal_(self.hidden_layer_2.weight, mean=0, std=0.01)
        # nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)
        # nn.init.normal_(self.mean_layer.weight, mean=0,std=0.01)
        # nn.init.normal_(self.covariance_layer.weight, mean=0,std=0.01)

        nn.init.uniform_(self.input_layer.weight, 0,1)
        nn.init.uniform_(self.hidden_layer_1.weight, 0,1)
        nn.init.uniform_(self.hidden_layer_2.weight, 0,1)
        nn.init.uniform_(self.output_layer.weight, 0, 1)
        nn.init.uniform_(self.mean_layer.weight, 0,1)
        nn.init.uniform_(self.covariance_layer.weight, 0,1)
    def initialize_biases(self) -> None:
        nn.init.constant_(self.input_layer.bias, 0)
        nn.init.constant_(self.hidden_layer_1.bias, 0)
        nn.init.constant_(self.hidden_layer_2.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.constant_(self.covariance_layer.bias, 0)



    #     TODO fix activation function it shouldnt give RELU instance
    def forward(self, state, log_std_min=-20, log_std_max=2) -> [torch.Tensor, torch.Tensor]:
        x = self.activation_function(self.input_layer(state))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        mean = self.mean_layer(x)
        # TODO: look if this is correct way of getting those values
        std_deviation = torch.clamp(self.covariance_layer(x), min=log_std_min, max=log_std_max)
        return mean, std_deviation
    def policy_evaluation(self, state) -> [torch.Tensor, torch.Tensor]:

        mean, standard_deviation = self.forward(state)

        # TODO: fix problem with squaring negative problem gives nan as result
        # standard deviation cannot be negative
        # degree of variation or spread in dataset
        # std_dev = square root of variance
        # variance is also nonegative because it is standard deviation squared

        # standard_deviation = torch.sqrt(torch.diag(covariance))

        basic_mean = torch.tensor([0.0])
        basic_std_dev = torch.tensor([1.0])


        gaussian_distribution = dist.Normal(basic_mean, basic_std_dev)
        noise = gaussian_distribution.sample()
        new_action_sampled_from_policy = torch.tanh(mean + standard_deviation*noise)
        # log probability is computed by approximating log-likelihood  of torch.tanh(mean + standard_deviation*noise)

        # right part is kinda easy
        # left part check probability density - there is function in pytorch that evaluates value of of prob. density function for u
        log_probability_of_new_action = (gaussian_distribution.log_prob(mean + standard_deviation*noise)
                                         - torch.log(1 - new_action_sampled_from_policy**2))
        return new_action_sampled_from_policy, log_probability_of_new_action
    def select_action(self, state) -> [torch.Tensor, torch.Tensor]:
        ...