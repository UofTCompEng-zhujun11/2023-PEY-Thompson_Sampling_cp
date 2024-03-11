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


class sim:
    
    def __init__(self, numArms) -> None:
        self.bandit_probs       = None
        self.numArms            = numArms
        self.wald_stats_FPR     = []
        self.wald_reject_FPR    = []


    def set_bandit_probs(self, probs) -> None:
        self.bandit_probs = probs
        assert(len(probs) == self.numArms)
        

    def get_vary_one_probs(self, amount) -> list:
        varied_probs = self.bandit_probs
        if amount > 0:
            result = min(self.bandit_probs[self.numArms - 1] + amount, 0.9999)
        else:
            result = max(self.bandit_probs[self.numArms - 1] + amount, 0.0001)
        varied_probs[self.numArms - 1] = result
        return varied_probs
    

    def get_vary_all_probs(self, amount) -> list:
        varied_probs = list()
        for index in range(self.numArms):
            if amount > 0:
                result = min(self.bandit_probs[index] + amount, 0.9999)
            else:
                result = max(self.bandit_probs[index] + amount, 0.0001)
            varied_probs.append(result)

        return varied_probs


