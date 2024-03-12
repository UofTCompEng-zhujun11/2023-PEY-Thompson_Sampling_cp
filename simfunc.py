from algrithms import NUM_ARMS
from algrithms import discounted_thompson_sampling
from algrithms import discounted_sliding_thompson_sampling
from algrithms import dts_alg_data
from algrithms import dsts_alg_data
from datas import simConfig
from datas import sim
from datas import TimeSteps
import numpy as np
import random
from datetime import datetime

random.seed(datetime.now().timestamp())
dts_sim_data = sim()
dsts_sim_data = sim()
config = simConfig(NUM_ARMS)

def simulation_fixed_vary(df, sw = None):
    global config

    for i in range(config.interval):
        if config.isvary_all:
            varied_probs = config.get_vary_all_probs(config.change_by * i)
        else:
            varied_probs = config.get_vary_one_probs(config.change_by * i)
        
        if config.alg == 'dts':
            discounted_thompson_sampling(varied_probs, config.sim_per_interval, df, (config.sim_per_interval*i))

        else:
            discounted_sliding_thompson_sampling(varied_probs, config.sim_per_interval, df, sw, (config.sim_per_interval*i))

    fpr(config.alg)
    power(config.alg)
    
    return

def simulation_rand_vary(df, sw = None):
    global config

    for i in range(config.interval):
        change_by = random.random()
        if config.isvary_all:
            varied_probs = config.get_vary_all_probs(change_by)
        else:
            varied_probs = config.get_vary_one_probs(change_by)
        
        if config.alg == 'dts':
            discounted_thompson_sampling(varied_probs, config.sim_per_interval, df, (config.sim_per_interval*i))

        else:
            discounted_sliding_thompson_sampling(varied_probs, config.sim_per_interval, df, sw, (config.sim_per_interval*i))

    fpr(config.alg)
    power(config.alg)
    
    return

def p_hat(alg_data:TimeSteps):
    p_hat = []

    for index in range(NUM_ARMS):
        # alg_data.arms[index].total_reward / alg_data.arms[index].total_reward
        p_hat.append(alg_data.arms[index].total_reward / alg_data.arms[index].total_pull)

    return p_hat

def fpr(alg):
    global dts_sim_data, dsts_alg_data

    if alg == 'dts':
        alg_data = dts_alg_data
        sim_data = dts_sim_data
    else:
        alg_data = dsts_alg_data
        sim_data = dsts_sim_data

    p_hat_list = []
    p_hat_list = p_hat(alg_data)
    p_hat_diff = p_hat_list[0] - p_hat_list[1]
    hyp_prob = 0.5*0.5
    arm_total_pull = []
    for index in range(NUM_ARMS):
        arm_total_pull.append(alg_data.arms[index].total_pull)

    arm_total_trials = (1/(arm_total_pull[0] - 1)) + (1/(arm_total_pull[1] - 1))

    sd = np.sqrt(hyp_prob * arm_total_trials)

    wald_stat = p_hat_diff/sd
    sim_data.wald_stats_FPR.append(wald_stat)
 
    reject = abs(wald_stat) > 1.96
    sim_data.wald_reject_FPR.append(reject)

def power(alg):
    global dts_sim_data, dsts_alg_data

    if alg == 'dts':
        alg_data = dts_alg_data
        sim_data = dts_sim_data
    else:
        alg_data = dsts_alg_data
        sim_data = dsts_sim_data

    p_hat_list = p_hat(alg_data)
    p_hat_diff = p_hat_list[0] - p_hat_list[1]

    var_1 = (p_hat_list[0] * (1 - p_hat_list[0])) / (alg_data.arms[0].total_pull - 1)
    var_2 = (p_hat_list[1] * (1 - p_hat_list[1])) / (alg_data.arms[1].total_pull - 1)

    sd = np.sqrt(var_1 + var_2)

    wald_stat = p_hat_diff/sd
    sim_data.wald_stats_power.append(wald_stat)

    reject = abs(wald_stat) > 1.96
    sim_data.wald_reject_power.append(reject)

    
def megasim(mega_trail, df, sw = None):
    global config, dts_sim_data, dsts_alg_data, dts_alg_data

    if config.alg == 'dts':
        alg_data = dts_alg_data
        sim_data = dts_sim_data
    else:
        alg_data = dsts_alg_data
        sim_data = dsts_sim_data


    for i in range(mega_trail):
        if config.isFixed:
            simulation_fixed_vary(df, sw)
        else:
            simulation_rand_vary(df, sw)
        if i == mega_trail - 1:
            sim_data.reward_ref = list(alg_data.arms[1].reward_record)
        dts_alg_data.clear_data()
        dsts_alg_data.clear_data()
        sim_data.set_overall()

    return