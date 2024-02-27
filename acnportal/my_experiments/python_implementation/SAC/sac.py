from q_policy_n import QNetwork
from policy_n import PolicyNetwork
from value_n import ValueNetwork
from replay_b import  ReplayBuffer
import torch
import gymnasium as gym
import torch.nn.functional as F
import torch.optim as optim


# TODO: change warnings filter this seems not to work
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead")



# rl algorithms are very brittle, with small change there can be the worst possible outcome

# better is to call this after each optimizer so i dont have to worry about cycling
# assuming we use only mse loss, if not, we can add parameter to the function
# def update_parameters(optimizer:torch.optim.Adam, outputs,targets):
#
#     loss_fn = torch.nn.MSELoss(outputs,targets)
#     # loss = loss_fn()
#
#
#     optimizer.zero_grad()
#     loss_fn.backward()
#     optimizer.step()




#         loss function must be differentiable to perform backward, compute gradients

# batch_size = 256

soft_target_update_tau = 0.005
hard_target_update_tau = 1
# two variants of SAC
# 1.hard target update: copies value function network weights every 1000 iterations instead of expontential smoothed
# average of weights
# 2. uses deterministic policy


# pendulum should learn under 7k steps

# new sac with updateable temperature performs at least as good as sac with constant temperature
# it matters when there is more exploration needed like in mountainous car
def update(buffer:ReplayBuffer, batch_size=256, discount_gamma=0.99) -> None:
    '''


    :param buffer:
    :param batch_size:
    :param discount_gamma:
    :return:

    done is a boolean flag with value 0 or 1
    # TODO:learning rate should be in update section or nearby specified somewhere
    # TODO: lookout where should be optimiser adam used and how
    # TODO: sample method must be implemented correctly


    # TODO: update should be perfomed in this order:
        1. update the value network param
        2. update the q networks 1,2 param
        3. update the policy network param
    '''
    state, action, reward, next_state, done = buffer.sample(batch_size=batch_size)

    # TODO: look at MBSE error in spinning up later on
    # Q training function
    # there were experiments and found that more than 2 Q functions only increase complexity, not performance
    predicted_q_value1 = q_network.forward(state, action)
    predicted_q_value2 = q_network2.forward(state, action)

    # predicted q value must be only a number it seems
    new_action, log_probability_of_new_action = policy_network.policy_evaluation(state)

    # V network training
    predicted_q_value_min = torch.min(predicted_q_value1, predicted_q_value2)
    target_v_value = predicted_q_value_min - log_probability_of_new_action
    value_loss = torch.nn.MSELoss(value_network.forward(state), target_v_value)
    # value_loss = F.mse_loss(value_network.forward(state), target_v_value)

    value_network_optimizer.zero_grad()
    value_loss.backward()
    value_network_optimizer.step()


    target_q_value = reward + discount_gamma * (1 - done) * target_value_network.forward(next_state)
    q_value_loss1 = torch.nn.MSELoss(predicted_q_value1, target_q_value)

    q_network_optimizer.zero_grad()
    q_value_loss1.backward()
    q_network_optimizer.step()

    q_value_loss2 = torch.nn.MSELoss(predicted_q_value2, target_q_value)

    q_network_optimizer2.zero_grad()
    q_value_loss2.backward()
    q_network_optimizer2.step()

    # q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value)
    # q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value)


    #   training policy
    policy_loss = torch.nn.MSELoss(log_probability_of_new_action, predicted_q_value_min)
    # policy_loss = F.mse_loss(log_probability_of_new_action, predicted_q_value_min)


    policy_network_optimizer.zero_grad()
    policy_loss.backward()
    policy_network_optimizer.step()



# test with simple environmnets first
# MountainCarContinuous-v0 environment







env = gym.make('MountainCarContinuous-v0')

replay_buffer = ReplayBuffer()
neurons_per_hidden_layer = 256

# did they specify how many iterations and env steps were there?
# difference between hard update and normal update?

# D is the distribution of previously sampled states and
# actions, or a replay buffer


# maju sa tam teraz dat premenne (stav a akcia) alebo az potom
# state musi byt numpy alebo od pytorch

state_dim = len(env.reset()[0])
# assuming that action space is one dimensional, if it is not, then it can be flattened
action_space = env.action_space
action_space_dim = action_space.shape[0]


value_network = ValueNetwork(state_dim=state_dim)
target_value_network = ValueNetwork(state_dim=state_dim)

q_network = QNetwork(state_dim=state_dim, action_dim=action_space_dim)
q_network2 = QNetwork(state_dim=state_dim, action_dim=action_space_dim)

policy_network = PolicyNetwork(state_dim=state_dim, action_dim=action_space_dim)



# TODO: check the documentation of adam, what problems it solves to understand
# TODO: test possible results with other optimisers

# TODO lookup all phases of NN in lectures, and also adam if there is explained or types of networks, more details
# FORWARD PASS through neural network to calculate loss function


# ADAM: minimize the loss function during the training of neural networks
# so for each loss function use adam to minimise loss

shared_lr_from_book = 3 * 10**-4

# HINT:gradients won’t be calculated for parameters, which are using requires_grad=False

value_network_optimizer = optim.Adam(value_network.parameters(),lr=shared_lr_from_book)
# target_value_network_optimizer = optim.Adam(value_network.parameters(),lr=shared_lr_from_book)

# TODO: one q optimizer or two : probably 2 since there are 2 loss functions
q_network_optimizer = optim.Adam(q_network.parameters(), lr=shared_lr_from_book)
q_network_optimizer2 = optim.Adam(q_network2.parameters(), lr=shared_lr_from_book)
policy_network_optimizer = optim.Adam(policy_network.parameters(), lr=shared_lr_from_book)






iterations = 10
# ([position, velocity],{}) {} is typically empty

# toto je dobre cislo
environment_steps = 1000
# zalezi od toho aky je update, predpokladajme ze je soft
gradient_steps = 1000



# minibatch learning
average_rewards = []


for i in range(iterations):
    # explain state of mountainous car v0

    # state consists of 2 variables
    # 1st variable: position of car along x-axis
    # 2nd variable: velocity of the car
    # action: acceleration of car in either direction
    # goal: to accelerate car to final state on right hill
    init_state = torch.from_numpy(env.reset()[0])
    average_reward = 0
    for env_step in range(environment_steps):
        action, _ = policy_network.policy_evaluation(init_state)
        # observation space gives weird type of result check again if it should match with state


        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
        # make sure order of elements is same as stored order in replay buffer for understanding

        # the actions and probabilities seem to be mapped correctly, because positive actions get higher probability values
        replay_buffer.add(state=init_state, action=action, reward=reward, next_state=next_state)
    for grad_step in range(gradient_steps):
        update(buffer=replay_buffer,
               batch_size=256,
               discount_gamma=0.99)