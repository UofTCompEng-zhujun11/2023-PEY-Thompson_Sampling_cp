import main
import Simulation
import matplotlib.pyplot as plt
import numpy as np

Simulation.simulation_vary_one([0.3, 0.4], 0.8, 500, 3, 0.02, 'dsts', 5)
# Simulation.simulation_vary_both([0.3, 0.4], 0.8, 500, 3, 0.02, 'dts')
# Simulation.simulation_vary_one([0.2, 0.3], 0.8, 500, 5, 0.02, 'dsts', 5)


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


reward_arm_1 = retrieve_reward_at_t(main.dts_full_rec, 0)
reward_arm_2 = retrieve_reward_at_t(main.dts_full_rec, 1)
power_arm_1 = retrieve_power_at_t(main.dts_power_at_t, 0)
power_arm_2 = retrieve_power_at_t(main.dts_power_at_t, 1)
fp_arm_1 = retrieve_false_positive_at_t(main.dts_false_positive_at_t, 0)
fp_arm_2 = retrieve_false_positive_at_t(main.dts_false_positive_at_t, 1)


reward_arm_1_dsts = retrieve_reward_at_t(main.dsts_full_rec, 0)
reward_arm_2_dsts = retrieve_reward_at_t(main.dsts_full_rec, 1)
power_arm_1_dsts = retrieve_power_at_t(main.dsts_power_at_t, 0)
power_arm_2_dsts = retrieve_power_at_t(main.dsts_power_at_t, 1)
fp_arm_1_dsts = retrieve_false_positive_at_t(main.dsts_false_positive_at_t, 0)
fp_arm_2_dsts = retrieve_false_positive_at_t(main.dsts_false_positive_at_t, 1)


x = np.linspace(0, 1000, 1499)
y1 = fp_arm_1_dsts
y2 = fp_arm_2_dsts

print(y1)


def plot_result(x, y1, y2, title, x_lab, y_lab):
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.grid(True)  # add gridlines
    plt.legend()  # add legend
    plt.show()


plot_result(x, y1, y2, 'reward comparision', 'reward_arm_1', 'reward_arm_2')

# x = np.linspace(0, 1000, 1499)
# y1 = reward_arm_1
# y2 = reward_arm_2
#
# y3 = power_arm_1
# y4 = power_arm_2
#
# y5 = fp_arm_1
# y6 = fp_arm_2

# print(fp_arm_2)
# # Plot the sine wave
# plt.plot(x, y1, label='arm 1 reward')
# plt.plot(x, y2, label='arm 2 reward')
# plt.xlabel('x')
# plt.ylabel('sin(x)')
# plt.grid(True)  # add gridlines
# plt.legend()  # add legend
# plt.show()


# Plotting result
# def plot_result(x, y1, y2, title, x_lab, y_lab):
#     plt.plot(x, y1, label='y1')
#     plt.plot(x, y2, label='y2')
#     plt.title(title)
#     plt.xlabel(x_lab)
#     plt.ylabel(y_lab)
#     plt.grid(True)  # add gridlines
#     plt.legend()  # add legend
#     plt.show()

