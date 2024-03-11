from algrithms import NUM_ARMS
from algrithms import discounted_thompson_sampling
from algrithms import discounted_sliding_thompson_sampling
from algrithms import dts_alg_data
from algrithms import dsts_alg_data
from datas import sim
from datas import TimeSteps
import numpy as np
import random
from datetime import datetime

random.seed(datetime.now().timestamp())
sim_data = sim(NUM_ARMS)

class simConfig:
    
    def __init__(self) -> None:
        self.isFixed            = False
        self.isvary_all         = False
        self.change_by          = None
        self.sim_per_interval   = None
        self.interval           = None
        self.alg                = ""

config = simConfig()

def simulation_fixed_vary(bandit_probs, df, sw = None):
    global sim_data, config

    sim_data.set_bandit_probs(bandit_probs)
    for i in range(config.interval):
        if config.isvary_all:
            varied_probs = sim_data.get_vary_all_probs(config.change_by * i)
        else:
            varied_probs = sim_data.get_vary_one_probs(config.change_by * i)
        
        if config.alg == 'dts':
            discounted_thompson_sampling(varied_probs, config.sim_per_interval, df, (config.sim_per_interval*i))

        else:
            discounted_sliding_thompson_sampling(varied_probs, config.sim_per_interval, df, sw, (config.sim_per_interval*i))

    fpr(config.alg)
    return

def simulation_rand_vary(bandit_probs, df, sw = None):
    global sim_data, config

    sim_data.set_bandit_probs(bandit_probs)
    for i in range(config.interval):
        change_by = random.random()
        if config.isvary_all:
            varied_probs = sim_data.get_vary_all_probs(change_by)
        else:
            varied_probs = sim_data.get_vary_one_probs(change_by)
        
        if config.alg == 'dts':
            discounted_thompson_sampling(varied_probs, config.sim_per_interval, df, (config.sim_per_interval*i))

        else:
            discounted_sliding_thompson_sampling(varied_probs, config.sim_per_interval, df, sw, (config.sim_per_interval*i))

    fpr(config.alg)
    return

def p_hat(alg_data:TimeSteps):
    p_hat = []

    for index in range(NUM_ARMS):
        num_trails = alg_data.get_num_trails()
        p_hat.append(alg_data.arms[index].total_pull/num_trails)

    return p_hat

def fpr(alg):
    global sim_data

    if alg == 'dts':
        alg_data = dts_alg_data
    else:
        alg_data = dsts_alg_data

    p_hat_list = []
    p_hat_list = p_hat(alg_data)
    p_hat_diff = p_hat_list[0] - p_hat_list[1]
    hyp_prob = 0.5*0.5
    arm_total_pull = []
    for index in range(NUM_ARMS):
        arm_total_pull.append(alg_data.arms[index].total_pull)

    arm_total_trials = (1/(arm_total_pull[0]) - 1)*(1/(arm_total_pull[1]) - 1)

    sd = np.sqrt(hyp_prob * arm_total_trials)

    wald_stat = p_hat_diff/sd
    sim_data.wald_stats_FPR.append(wald_stat)
 
    reject = abs(wald_stat) > 1.96
    sim_data.wald_reject_FPR.append(reject)

def power():
    global sim_data

    
def megasim(mega_trail, bandit_probs, df, sw = None):
    global config

    for i in range(mega_trail):
        if config.isFixed:
            simulation_fixed_vary(bandit_probs, df, sw)
        else:
            simulation_rand_vary(bandit_probs, df, sw)

