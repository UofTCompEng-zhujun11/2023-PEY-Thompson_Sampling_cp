import main
import Simulation
import matplotlib.pyplot as plt
import numpy as np

# Best arm remains unchanged, but continue to increase in mean reward
Simulation.simulation_vary_one([0.3, 0.4], 0.8, 500, 3, 0.02, 'dsts', 5)
Simulation.simulation_vary_one([0.3, 0.4], 0.8, 500, 3, 0.02, 'dts')

# Best arm continue to increase in mean reward
Simulation.simulation_vary_one([0.5, 0.9], 0.8, 500, 3, -0.02, 'dsts', 5)
Simulation.simulation_vary_one([0.5, 0.9], 0.8, 500, 3, -0.02, 'dts')


# Functions to compute the criteria
def retrieve_reward_at_t(info, arm):
    accumulated_rew = 0
    reward_at_t = []
    for t in info:
        accumulated_rew += t[arm]
        reward_at_t.append(accumulated_rew)

    return reward_at_t


def retrieve_power_at_t(info, arm):
    power_at_t = []
    for t in info:
        power_at_t.append(t[arm])

    return power_at_t


def retrieve_false_positive_at_t(info, arm):
    fp_at_t = []
    for t in info:
        fp_at_t.append(t[arm])

    return fp_at_t


def put_tgt(prop):
    if prop == 'reward':
        prop_arm_1_dts = retrieve_reward_at_t(main.dts_full_rec, 0)
        prop_arm_2_dts = retrieve_reward_at_t(main.dts_full_rec, 1)
        prop_arm_1_dsts = retrieve_reward_at_t(main.dsts_full_rec, 0)
        prop_arm_2_dsts = retrieve_reward_at_t(main.dsts_full_rec, 1)

    if prop == 'power':
        prop_arm_1_dts = retrieve_power_at_t(main.dts_power_at_t, 0)
        prop_arm_2_dts = retrieve_power_at_t(main.dts_power_at_t, 1)
        prop_arm_1_dsts = retrieve_power_at_t(main.dsts_power_at_t, 0)
        prop_arm_2_dsts = retrieve_power_at_t(main.dsts_power_at_t, 1)

    else:
        prop_arm_1_dts = retrieve_false_positive_at_t(main.dts_false_positive_at_t, 0)
        prop_arm_2_dts = retrieve_false_positive_at_t(main.dts_false_positive_at_t, 1)
        prop_arm_1_dsts = retrieve_false_positive_at_t(main.dsts_false_positive_at_t, 0)
        prop_arm_2_dsts = retrieve_false_positive_at_t(main.dsts_false_positive_at_t, 1)

    return [prop_arm_1_dts, prop_arm_2_dts, prop_arm_1_dsts, prop_arm_2_dsts]


x = np.linspace(0, 1000, 1500)

y = put_tgt('reward')


def plot_result(x, y, title, x_lab, y_lab, to_be_plot):
    for i in y:
        plt.plot(x, i, label=str(to_be_plot) + " for arm" + str(1))

    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid(True)  # add gridlines
    plt.legend()  # add legend
    plt.show()


plot_result(x, y, 'False Positive Rates Comparison Between the Two Algorithms',
            'Time Steps', 'False Positive Rates', 'False Positive Rates')



