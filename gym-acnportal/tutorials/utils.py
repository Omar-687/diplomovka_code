import matplotlib.pyplot as plt
import numpy as np
# num_of_timesteps_per_episode must be constant, for understanding other visualisations better
# plots price according tonxgin li article, 1 - t/24
def plot_daily_price(period=12):
    # x_values = np.arange(0,num_of_timesteps_per_episode + 1,1)
    time_interval = period/60
    time = 0
    y_values = []
    x_values = []
    while time <= 24:
        x_values.append(time)
        # specify cost function
        y_values.append(1 - time/24)
        time += time_interval

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)

    ax.set(xlabel='Time (hours)', ylabel='Price ($)',
           title='Price for 1kWh of energy.')
    # plt.xlim([0, 24])
    # ax.grid()
    xticks = np.arange(0,24 + 1,2)
    ax.set_xticks(xticks)
    plt.show()
def plot_substation_rates_given_by_operator(signals, period=12):
    time_interval = period / 60
    time = 0
    x_values = []
    while time <= 24:
        x_values.append(time)
        time += time_interval
#     TODO: fix case if signals is shorter than time, for example if there are no evs in given day
    if len(x_values) != len(signals):
        raise ValueError('x axis and y axis must have same length!')
    fig, ax = plt.subplots()
    ax.plot(x_values, signals)

    ax.set(xlabel='Time (hours)', ylabel='Substation rate (kWh)',
           title='Generated substation rates from PPC.')
    # plt.xlim([0, 24])
    # ax.grid()
    xticks = np.arange(0, 24 + 1, 2)
    ax.set_xticks(xticks)
    plt.show()

plot_daily_price()
plot_substation_rates_given_by_operator(120*[10], period=12)