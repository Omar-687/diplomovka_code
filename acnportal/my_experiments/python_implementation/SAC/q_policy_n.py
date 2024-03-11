import torch
from torch import nn

# standard
class QNetwork(nn.Module):
    # shared
    # 2 hidden layers
    # relu
    # hidden units per layer
    # in mountain car continuous the state dim is 2 and action dim is 1
    def __init__(self, state_dim, action_dim, neurons_per_hidden_layer=256):
        super(QNetwork, self).__init__()
        self.activation_function = torch.nn.ReLU()
        self.input_layer = nn.Linear(in_features=state_dim+action_dim, out_features=neurons_per_hidden_layer)
        self.hidden_layer_1 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.hidden_layer_2 = nn.Linear(in_features=neurons_per_hidden_layer, out_features=neurons_per_hidden_layer)
        self.output_layer = nn.Linear(in_features=neurons_per_hidden_layer, out_features=1)

        self.initialize_weights()
        self.initialize_biases()


    def initialize_weights(self) -> None:
        # nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.hidden_layer_1.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.hidden_layer_2.weight, mean=0.0, std=0.01)
        # nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.uniform_(self.input_layer.weight,-3e-3,3e-3)
        nn.init.uniform_(self.hidden_layer_1.weight,-3e-3,3e-3)
        nn.init.uniform_(self.hidden_layer_2.weight,-3e-3,3e-3)
        nn.init.uniform_(self.output_layer.weight,-3e-3,3e-3)
    def initialize_biases(self) -> None:
        nn.init.constant_(self.input_layer.bias, 0)
        nn.init.constant_(self.hidden_layer_1.bias, 0)
        nn.init.constant_(self.hidden_layer_2.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)
    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x