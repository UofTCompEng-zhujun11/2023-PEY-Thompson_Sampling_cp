import numpy as np
from datas import TimeSteps

NUM_ARMS = 2

dts_alg_data = TimeSteps(NUM_ARMS)
dsts_alg_data = TimeSteps(NUM_ARMS)

def discounted_thompson_sampling(bandit_probs, n, df, startp=0):
    """
    bandit_probs: An array of probability of reward for each arm in order
    record: record of the choice of arms at time t
    n: The number of trials
    df: Discount factor
    startp: Trial staring point
    returns nothing
    """

    # global dts_choice_rec, dts_pulls_total, dts_reward_total, dts_full_rec
    global dts_alg_data

    # Picking arms
    for i in range(startp, startp + n):
        arm_0_prob = dts_alg_data.get_arm_prob(df, 0)
        arm_1_prob = dts_alg_data.get_arm_prob(df, 1)

        predicted_prob = [arm_0_prob, arm_1_prob]
        choice = np.argmax(predicted_prob)

        # Generate reward based on the choice of arms
        isReward = np.random.choice([0, 1], p=(1 - bandit_probs[choice], bandit_probs[choice]))
        for arm_index in range(NUM_ARMS):
            if arm_index == choice:
                dts_alg_data.arms[arm_index].pull_reward(isReward, True)
            else:
                dts_alg_data.arms[arm_index].pull_reward(False, False)


    return

def discounted_sliding_thompson_sampling(bandit_probs, n, df, sw, startp=0):
    """
    bandit_probs: An array of probability of reward for each arm in order
    record: record of the choice of arms at time t
    n: The number of trials
    df: Discount factor
    sw: Sliding window
    startp: Trial staring point
    return nothing
    """

    global dsts_alg_data

    # Picking arms, alter probability measure based on how old the info is
    for i in range(startp, startp + n):
        if i >= sw:
            arm_0_prob = dsts_alg_data.get_arm_prob_old(df, 0, (i - sw), i)
            arm_1_prob = dsts_alg_data.get_arm_prob_old(df, 1, (i - sw), i)
            arm_0_hot_prob = dsts_alg_data.get_arm_prob_hot(0, (i - sw), i)
            arm_1_hot_prob = dsts_alg_data.get_arm_prob_hot(1, (i - sw), i)

        else:
            arm_0_prob = dsts_alg_data.get_arm_prob(1, 0)
            arm_1_prob = dsts_alg_data.get_arm_prob(1, 1)
            arm_0_hot_prob = arm_0_prob
            arm_1_hot_prob = arm_1_prob

        predicted_prob = [arm_0_prob, arm_1_prob, arm_0_hot_prob, arm_1_hot_prob]
        choice = np.argmax(predicted_prob) % NUM_ARMS

        # Generate reward based on the choice of arms
        isReward = np.random.choice([0, 1], p=(1 - bandit_probs[choice], bandit_probs[choice]))
        for arm_index in range(NUM_ARMS):
            if arm_index == choice:
                dsts_alg_data.arms[arm_index].pull_reward(isReward, True)
            else:
                dsts_alg_data.arms[arm_index].pull_reward(False, False)

    return