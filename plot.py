from simfunc import dts_sim_data
from simfunc import dsts_sim_data
from simfunc import config
from simfunc import megasim
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
    if criteria == 'Reward':
        result_dts = retrieve_reward_at_t(dts_sim_data.reward_ref)
        result_dsts = retrieve_reward_at_t(dsts_sim_data.reward_ref)
    elif criteria == 'Power':
        result_dts = dts_sim_data.overall_power
        result_dsts = dsts_sim_data.overall_power
    else:
        result_dts = dts_sim_data.overall_FPR
        result_dsts = dsts_sim_data.overall_FPR

    return result_dts, result_dsts
        
def plot_result(x, y, title, x_lab, y_lab, to_be_plot):
    counter = 0
    for i in y:
        plt.plot(x, i, label=str(to_be_plot) + " " + ("dts" if counter < 1 else "dsts"))
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
    config.interval = 1
    config.sim_per_interval = 100
    config.change_by = 0
    config.isvary_all = False
    config.bandit_probs = [0.5, 0.5]

    megasim(1000, 1)
    config.alg = "dsts"
    megasim(1000, 1, 20)

    x = np.linspace(0, 1000, 1000)
    y = compile_info('FPR')
    plot_result(x, y, 'Comparison Between the Two Algorithms FPR',
            'Time Steps', 'FPR', 'FPR')

    return

constructPlot()
# Best arm decreases below other arm, small change, quickly
config.alg = "dts"
config.isFixed = True
config.interval = 10
config.sim_per_interval = 100
config.change_by = - 0.02
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# Best arm decreases below other arm, large change, slowly
config.alg = "dts"
config.isFixed = True
config.interval = 3
config.sim_per_interval = 500
config.change_by = -0.1
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# Sub-optimal arm increases over best-arm, large change, slowly
config.alg = "dts"
config.isFixed = True
config.interval = 3
config.sim_per_interval = 500
config.change_by = 0.1
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# Sub-optimal arm increases over best-arm, small change, quickly
config.alg = "dts"
config.isFixed = True
config.interval = 10
config.sim_per_interval = 100
config.change_by = - 0.02
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# Sub-optimal arm suddenly increase over best, large interval
config.alg = "dts"
config.isFixed = True
config.interval = 2
config.sim_per_interval = 500
config.change_by = 0.3
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# Sub-optimal arm suddenly increase over best, small interval
config.alg = "dts"
config.isFixed = True
config.interval = 2
config.sim_per_interval = 100
config.change_by = 0.3
config.isvary_all = False
config.bandit_probs = [0.5, 0.6]

# arms randomly varying
config.alg = "dts"
config.isFixed = False
config.interval = 5
config.sim_per_interval = 200
config.change_by = 0.3
config.isvary_all = True
config.bandit_probs = [0.5, 0.6]

# both arm stationary
config.alg = "dts"
config.isFixed = True
config.interval = 5
config.sim_per_interval = 100
config.change_by = 0
config.isvary_all = True
config.bandit_probs = [0.5, 0.6]