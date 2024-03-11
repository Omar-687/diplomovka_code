import torch
from torch import nn
# NEURAL NETWORK
# input layer hidden layer output layer
# Pytorch
# we use this neural netowrk to approximate value of value function



# expected sum of rewards


# TODO:expressive NN: what is the meaning behind it?
class ValueNetwork(nn.Module):
    # Its flow is uni - directional, meaning that
    # the
    # information in the
    # model
    # flows in only
    # one
    # direction—forward—from the input
    # nodes, through
    # the
    # hidden
    # nodes( if any) and to
    # the
    # output
    # nodes, without
    # any
    # cycles or loops
    def __init__(self, state_dim, neurons_per_hidden_layer=256):
        super(ValueNetwork, self).__init__()
        self.activation_function = nn.ReLU()
        # not sure if there should be linear layer
        # do i need to change it or

        # bias and weights?

        self.input_layer = nn.Linear(in_features=state_dim, out_features= neurons_per_hidden_layer)
        self.hidden_layer_1 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.hidden_layer_2 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.output_layer = nn.Linear(in_features=neurons_per_hidden_layer, out_features=1)

        self.initialize_weights()
        self.initialize_biases()


    # try with simpler tasks, if it works with bigger keep this bias otherwise change
    # bias value is not directly mentioned in the book
    # 3*10^-3 they used
    def initialize_weights(self) -> None:
        # nn.init.normal_(self.input_layer.weight, mean=0.0, std=1)
        # nn.init.normal_(self.hidden_layer_1.weight, mean=0.0, std=1)
        # nn.init.normal_(self.hidden_layer_2.weight, mean=0.0, std=1)
        # nn.init.normal_(self.output_layer.weight, mean=0.0, std=1)

        # nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.hidden_layer_1.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.hidden_layer_2.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.uniform_(self.input_layer.weight, 0, 1)
        nn.init.uniform_(self.hidden_layer_1.weight, 0, 1)
        nn.init.uniform_(self.hidden_layer_2.weight, 0, 1)
        nn.init.uniform_(self.output_layer.weight, 0, 1)
    def initialize_biases(self) -> None:
        nn.init.constant_(self.input_layer.bias, 0)
        nn.init.constant_(self.hidden_layer_1.bias, 0)
        nn.init.constant_(self.hidden_layer_2.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)
    #     bias


    def forward(self, state:torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(state))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x