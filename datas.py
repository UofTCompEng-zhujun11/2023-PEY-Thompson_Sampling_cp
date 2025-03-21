import numpy as np

class arm:

    def __init__(self) -> None:
        self.total_pull         = 0
        self.total_reward       = 0
        self.pull_record        = list()
        self.reward_record      = list()

    def get_total(self) -> None:
        assert(len(self.pull_record) == len(self.reward_record))
        for time in range(len(self.pull_record)):
            self.total_pull += self.pull_record[time]
            self.total_reward += self.reward_record[time]

    def pull_reward(self, isReward, isPull) -> None:
        if isPull:
            self.pull_record.append(1)
            self.total_pull += 1
        else:
            self.pull_record.append(0)
        if isReward:
            self.reward_record.append(1)
            self.total_reward += 1
        else:
            self.reward_record.append(0)

    def clear_data(self):
        self.total_pull = 0
        self.total_reward = 0
        self.pull_record.clear()
        self.reward_record.clear()
    



class TimeSteps:

    def __init__(self, numArms) -> None:
        self.arms               = list()
        for count in range(numArms):
            self.arms.append(arm())


    def get_arm_prob(self, df, arm_index) -> float:
        discounted_reward = self.arms[arm_index].total_reward * df
        discounted_no_reward = (self.arms[arm_index].total_pull - self.arms[arm_index].total_reward) * df

        assert(discounted_no_reward >= 0 and discounted_reward >= 0)

        return np.random.beta(discounted_reward + 1, discounted_no_reward + 1)
    
    
    def get_arm_prob_old(self, df, arm_index, start, end) -> float:
        hot_total_pull, hot_total_reward = self.get_arm_hot_total(start, end, arm_index)
        old_total_pull = self.arms[arm_index].total_pull - hot_total_pull
        old_total_reward = self.arms[arm_index].total_reward - hot_total_reward

        discounted_reward = old_total_reward * df
        discounted_no_reward = (old_total_pull - old_total_reward) * df

        assert(discounted_no_reward >= 0 and discounted_reward >= 0)

        return np.random.beta(discounted_reward + 1, discounted_no_reward + 1)
    
    
    def get_arm_prob_hot(self, arm_index, start, end) -> float:
        hot_total_pull, hot_total_reward = self.get_arm_hot_total(start, end, arm_index)
        
        return np.random.beta(hot_total_reward + 1, (hot_total_pull - hot_total_reward) + 1)
    
    
    def get_arm_hot_total(self, start, end, arm_index):
        hot_total_pull = 0
        hot_total_reward = 0

        for steps in range(start, end):
            hot_total_pull += self.arms[arm_index].pull_record[steps]
            hot_total_reward += self.arms[arm_index].reward_record[steps]

        return hot_total_pull, hot_total_reward
    
    def get_num_trails(self):
        return len(self.arms[0].pull_record)
    
    def clear_data(self):
        for arm in self.arms:
            arm.clear_data()
    


class sim:
    
    def __init__(self) -> None:
        self.wald_stats_FPR     = []
        self.wald_reject_FPR    = []
        self.wald_stats_power   = []
        self.wald_reject_power  = []
        self.overall_power      = []
        self.overall_FPR        = []
        self.reward_ref         = []

    def set_overall(self) -> float:
        self.overall_FPR.append(sum(self.wald_reject_FPR)/len(self.wald_reject_FPR)) 
        self.overall_power.append(sum(self.wald_reject_power)/len(self.wald_reject_power))


class simConfig:
    
    def __init__(self, numArms) -> None:
        self.isFixed            = False
        self.isvary_all         = False
        self.change_by          = None
        self.sim_per_interval   = None
        self.interval           = None
        self.alg                = ""

        self.bandit_probs       = None

    def get_vary_one_probs(self, amount) -> list:
        varied_probs = list(self.bandit_probs)
        if amount > 0:
            result = min(self.bandit_probs[len(self.bandit_probs) - 1] + amount, 0.9999)
        else:
            result = max(self.bandit_probs[len(self.bandit_probs) - 1] + amount, 0.0001)
        varied_probs[len(self.bandit_probs) - 1] = result
        return varied_probs
    

    def get_vary_all_probs(self, amount) -> list:
        varied_probs = list()
        for index in range(len(self.bandit_probs)):
            if amount > 0:
                result = min(self.bandit_probs[index] + amount, 0.9999)
            else:
                result = max(self.bandit_probs[index] + amount, 0.0001)
            varied_probs.append(result)

        return varied_probs

