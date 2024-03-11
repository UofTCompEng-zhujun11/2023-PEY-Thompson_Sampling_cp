from simfunc import dts_sim_data
from simfunc import dsts_sim_data
from simfunc import config
from simfunc import megasim
from algrithms import dts_alg_data
from algrithms import dsts_alg_data
import matplotlib.pyplot as plt
import numpy as np


def retrieve_reward_at_t(info):
    accumulated_rew = 0
    reward_at_t = []
    for t in info:
        accumulated_rew += t
        reward_at_t.append(accumulated_rew)

    return reward_at_t


def compile_info (criteria):
    if criteria == 'reward':
        result_dts = retrieve_reward_at_t(dts_alg_data.arms[1].reward_record)
        result_dsts = retrieve_reward_at_t(dsts_alg_data.arms[1].reward_record)
    elif criteria == 'power':
        result_dts = dts_sim_data.overall_power
        result_dsts = dsts_sim_data.overall_power
    else:
        result_dts = dts_sim_data.overall_FPR
        result_dsts = dsts_sim_data.overall_FPR

    return result_dts, result_dsts
        
def plot_result(x, y, title, x_lab, y_lab, to_be_plot):
    counter = 0
    for i in y:
        plt.plot(x, i, label=str(to_be_plot) + " for arm " + str(counter % 2 + 1) + " " + ("dts" if counter < 2 else "dsts"))
        counter += 1

    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid(True)  # add gridlines
    plt.legend()  # add legend
    plt.show()

def constructPlot():
    config.alg = "dts"
    config.isFixed = True
    config.interval = 3
    config.sim_per_interval = 500
    config.change_by = 0.02
    config.isvary_all = False

    megasim(1000, [0.6, 0.65], 0.8)

    x = np.linspace(0, 1000, 1500)
    y = compile_info('reward')
    plot_result(x, y, 'False Positive Rates Comparison Between the Two Algorithms',
            'Time Steps', 'False Positive Rates', 'False Positive Rates')
    return

constructPlot()