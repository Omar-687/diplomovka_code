import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def plot_substation_rates_given_by_operator(signals, period=12, start_date="", end_date=""):
    time_interval = period / 60
    time = 0
    x_values = []
    while time <= 24:
        x_values.append(time)
        time += time_interval
#     TODO: fix case if signals is shorter than time, for example if there are no evs in given day
    if len(x_values) != len(signals):
        raise ValueError(f'x axis and y axis must have same length! {len(x_values)} != {len(signals)}')
    fig, ax = plt.subplots()
    ax.plot(x_values, signals)

    ax.set(xlabel='Time (hours)', ylabel='Substation rate (kWh)',
           title=f'Generated substation rates from PPC. ({start_date} - {end_date})')
    # plt.xlim([0, 24])
    # ax.grid()
    xticks = np.arange(0, 24 + 1, 2)
    ax.set_xticks(xticks)


    plt.show()


def plot_charging_rate_over_time(charging_rates, period=12, start_date="", end_date=""):
    time_interval = period / 60
    time = 0
    x_values = []
    while time <= 24:
        x_values.append(time)
        time += time_interval
    #     TODO: fix case if signals is shorter than time, for example if there are no evs in given day
    if len(x_values) != len(charging_rates):
        raise ValueError(f'x axis and y axis must have same length! {len(x_values)},{len(charging_rates)}')
    fig, ax = plt.subplots()
    ax.plot(x_values, charging_rates)

    ax.set(xlabel='Time (hours)', ylabel='Substation rate (kWh)',
           title=f'Substation rates given by Aggregator to EVs ({start_date} - {end_date})')
    # plt.xlim([0, 24])
    # ax.grid()
    xticks = np.arange(0, 24 + 1, 2)
    ax.set_xticks(xticks)
    plt.show()

def plot_mse_over_time():
    ...
def plot_mpe_over_time():
    ...



def plot_arrivals(events, period, start_date,end_date):
    time_interval = period / 60
    time = 0
    x_values = {}

    # 24*60 / period 121 steps
    for i in range(0,(24*60//period) + 1):
        x_values[i] = 0
    # while time <= 24:
    #     x_values[time] = 0
    #     time += time_interval

    for event in events:
        x_values[event[0]] += 1



    fig, ax = plt.subplots()
    ax.plot(x_values.keys(), x_values.values())
    #
    #
    #
    #

    ax.set(xlabel='Time (hours)', ylabel='Number of arrivals',
           title=f'Arrivals of EVs ({start_date} - {end_date})')

    xticks = np.arange(0, (24*60//period) + 1, 120/period)

    xticks_labels = np.arange(0, 24 + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)

    plt.show()


def plot_departures(events, period, start_date,end_date):
    time_interval = period / 60
    time = 0
    x_values = {}

    # 24*60 / period 121 steps
    for i in range(0,(24*60//period) + 1):
        x_values[i] = 0
    # while time <= 24:
    #     x_values[time] = 0
    #     time += time_interval

    for event in events:
        if event[1].ev.departure not in x_values.keys():
            continue
        x_values[event[1].ev.departure] += 1



    fig, ax = plt.subplots()
    ax.plot(x_values.keys(), x_values.values())
    #
    #
    #
    #

    ax.set(xlabel='Time (hours)', ylabel='Number of departures',
           title=f'Departures of EVs ({start_date} - {end_date})')

    xticks = np.arange(0, (24*60//period) + 1, 120/period)

    xticks_labels = np.arange(0, 24 + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)

    plt.show()

# overall energy charged
# overall requested energy
# percent of energy requested satisfied
# total price for charging
# demand charge implement in future if using tarrif costs as well

# this table is for one episode day primarily, but can be used for multiple days
def plot_table_for_other_info(overall_energy_charged,
                              overall_energy_requested,
                              number_of_evs,
                              total_charging_costs,
                              start_date,
                              end_date):
    d = {
    '% of delivered energy': [0 if overall_energy_requested==0 else
                              100 * (overall_energy_charged / overall_energy_requested)],
    'overall energy charged': [overall_energy_charged],
    'overall energy requested': [overall_energy_requested],
    #     one EV can arrive multiple times at charging station
    'Number of EVs arrivals': [number_of_evs],
    'total charging costs': [total_charging_costs],
    'start date': [start_date],
    'end date': [end_date],}
    df = pd.DataFrame(data=d)
    # Set option to display all rows
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

# plot_daily_price()
# plot_substation_rates_given_by_operator(120*[10], period=12)
# plot_table_for_other_info(0,0,0,0,0)

# def plot_departures(events):
#
