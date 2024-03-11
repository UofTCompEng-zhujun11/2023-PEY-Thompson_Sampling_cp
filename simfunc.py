from algrithms import NUM_ARMS
from algrithms import discounted_thompson_sampling
from algrithms import discounted_sliding_thompson_sampling
from algrithms import dts_alg_data
from algrithms import dsts_alg_data
from datas import sim
import numpy as np

sim_data = sim(NUM_ARMS)

def simulation(vary_all, bandit_probs, df, sim_per_interval, interval, change_by, alg, sw = None):
    global sim_data

    sim_data.set_bandit_probs(bandit_probs)
    for i in range(interval):
        if vary_all:
            varied_probs = sim_data.get_vary_all_probs(change_by * i)
        else:
            varied_probs = sim_data.get_vary_one_probs(change_by * i)
        
        if alg == 'dts':
            discounted_thompson_sampling(varied_probs, sim_per_interval, df, (sim_per_interval*i))

        else:
            discounted_sliding_thompson_sampling(varied_probs, sim_per_interval, df, sw, (sim_per_interval*i))

    fpr('dts')
    fpr('dsts')
    return

def p_hat():
    p_hat = []

    for index in range(NUM_ARMS):
        num_pulls = len(dts_alg_data.arms[index].pull_record)
        p_hat.append(dts_alg_data.arms[index].total_pull/num_pulls)

    return p_hat

def fpr(alg):
    global sim_data

    if alg == 'dts':
        data = dts_alg_data
    else:
        data = dsts_alg_data

    p_hat_list = p_hat()
    p_hat_diff = p_hat_list[0] - p_hat_list[1]
    hyp_prob = 0.5*0.5

    num_pulls = []
    for index in range(NUM_ARMS):
        num_pulls.append(len(data.arms[index].pull_record))

    arm_total_trials = (1/(num_pulls[0]) - 1)*(1/(num_pulls[1]) - 1)

    sd = np.sqrt(hyp_prob * arm_total_trials)

    wald_stat = p_hat_diff/sd
    sim_data.wald_stats_FPR.append(wald_stat)
 
    reject = abs(wald_stat) > 1.96
    sim_data.wald_reject_FPR.append(reject)

def power(alg):
    global sim_data

    if alg == 'dts':
        data = dts_alg_data
    else:
        data = dsts_alg_data

    p_hat_list = p_hat(data)
    p_hat_diff = p_hat_list[0] - 0.5

    var_1 = (p_hat_list[0] * (1 - p_hat_list[0])) / (data.arms[0].total_pull - 1)
    var_2 = (p_hat_list[1] * (1 - p_hat_list[1])) / (data.arms[1].total_pull - 1)

    sd = np.sqrt(var_1 + var_2)

    wald_stat = p_hat_diff/sd
    sim_data.wald_stats_power.append(wald_stat)

    reject = abs(wald_stat) > 1.96
    sim_data.wald_reject_power.append(reject)


    



    


simulation(True, [0.65, 0.7], 0.8, 500, 3, -0.2, 'dts')
simulation(True, [0.3, 0.7], 0.8, 500, 3, 0.08, 'dsts', sw=10)