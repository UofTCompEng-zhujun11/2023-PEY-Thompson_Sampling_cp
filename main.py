import numpy as np

#global variable defines(initialize the arm's distribution)

dsts_reward_rec = list(list())
dsts_pulls_rec = list(list())
dsts_reward_total = [0, 0]
dsts_pulls_total = [0, 0]

dsts_full_rec = list(list())

dts_choice_rec = list()
dts_reward_total = [0, 0]
dts_pulls_total = [0, 0]

dts_full_rec = list(list())


def discounted_thompson_sampling(bandit_probs, n, df, startp=0):
    """
    bandit_probs: An array of probability of reward for each arm in order
    record: record of the choice of arms at time t
    n: The number of trials
    df: Discount factor
    startp: Trial staring point
    returns nothing
    """

    global dts_choice_rec, dts_pulls_total, dts_reward_total, dts_full_rec

    # Picking arms
    for i in range(startp, startp + n):
        arm_1 = np.random.beta((dts_reward_total[0])*df + 1,
                               (dts_pulls_total[0] - dts_reward_total[0])*df + 1)
        arm_2 = np.random.beta((dts_reward_total[1])*df + 1,
                               (dts_pulls_total[1] - dts_reward_total[1])*df + 1)
        predicted_prob = [arm_1, arm_2]
        choice = np.argmax(predicted_prob)
        dts_choice_rec.append(choice)

        # Generate reward based on the choice of arms
        if choice == 0:
            a_pull = np.random.choice([0, 1], p=(1 - bandit_probs[0], bandit_probs[0]))

            if a_pull == 1:
                dts_reward_total[0] += 1
                dts_pulls_total[0] += 1
            else:
                dts_pulls_total[0] += 1

        else:
            b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))

            if b_pull == 1:
                dts_reward_total[1] += 1
                dts_pulls_total[1] += 1
            else:
                dts_pulls_total[1] += 1

    return


def discounted_thompson_sampling_for_sim(bandit_probs, n, df, startp=0):
    """
    bandit_probs: An array of probability of reward for each arm in order
    record: record of the choice of arms at time t
    n: The number of trials
    df: Discount factor
    startp: Trial staring point
    returns nothing
    """

    global dts_choice_rec, dts_pulls_total, dts_reward_total, dts_full_rec

    # Picking arms
    for i in range(startp, startp + n):
        arm_1 = np.random.beta((dts_reward_total[0])*df + 1,
                               (dts_pulls_total[0] - dts_reward_total[0])*df + 1)
        arm_2 = np.random.beta((dts_reward_total[1])*df + 1,
                               (dts_pulls_total[1] - dts_reward_total[1])*df + 1)
        predicted_prob = [arm_1, arm_2]
        choice = np.argmax(predicted_prob)
        dts_choice_rec.append(choice)

        # Generate reward based on the choice of arms
        if choice == 0:
            pull(0, bandit_probs, dts_reward_total, dts_pulls_total, dts_full_rec, 'dts')
        else:
            pull(1, bandit_probs, dts_reward_total, dts_pulls_total, dts_full_rec, 'dts')

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

    global dsts_reward_rec, dsts_pulls_rec, dsts_reward_total, dsts_pulls_total

    # Picking arms, alter probability measure based on how old the info is
    for i in range(startp, startp + n):
        if i >= sw:
            reward_new, pulls_new = sum((i-sw), i, dsts_reward_rec, dsts_pulls_rec)

            rewe_old1 = dsts_reward_total[0] - reward_new[0]
            pulls_old1 = dsts_pulls_total[0] - pulls_new[0]
            rewe_old2 = dsts_reward_total[1] - reward_new[1]
            pulls_old2 = dsts_pulls_total[1] - pulls_new[1]

            arm_1 = np.random.beta(rewe_old1*df + 1,
                                   (pulls_old1 - rewe_old1) * df + 1)
            arm_2 = np.random.beta(rewe_old2*df + 1,
                                   (pulls_old2 - rewe_old2) * df + 1)
        else:
            reward_new, pulls_new = sum(0, i, dsts_reward_rec, dsts_pulls_rec)
            arm_1 = np.random.beta(1, 1)
            arm_2 = np.random.beta(1, 1)

        arm_1_hot = np.random.beta(reward_new[0] + 1,
                                   (pulls_new[0] - reward_new[0]) + 1)
        arm_2_hot = np.random.beta(reward_new[1] + 1,
                                   (pulls_new[1] - reward_new[1]) + 1)

        predicted_prob = [arm_1, arm_2, arm_1_hot, arm_2_hot]
        choice = np.argmax(predicted_prob)

        # Generate reward based on the choice of arms
        if choice == 0:
            a_pull = np.random.choice([0, 1], p=(1 - bandit_probs[0], bandit_probs[0]))

            if a_pull == 1:
                dsts_reward_total[0] += 1
                dsts_pulls_total[0] += 1
                dsts_reward_rec.append([1, 0])
                dsts_pulls_rec.append([1, 0])

            else:
                dsts_pulls_total[0] += 1
                dsts_reward_rec.append([0, 0])
                dsts_pulls_rec.append([1, 0])

        else:
            b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))
            if b_pull == 1:
                dsts_reward_total[1] += 1
                dsts_pulls_total[1] += 1
                dsts_reward_rec.append([0, 1])
                dsts_pulls_rec.append([0, 1])
            else:
                dsts_pulls_total[1] += 1
                dsts_reward_rec.append([0, 0])
                dsts_pulls_rec.append([0, 1])
    return


def discounted_sliding_thompson_sampling_for_sim(bandit_probs, n, df, sw, startp=0):
    """
    bandit_probs: An array of probability of reward for each arm in order
    record: record of the choice of arms at time t
    n: The number of trials
    df: Discount factor
    sw: Sliding window
    startp: Trial staring point
    return nothing
    """

    global dsts_reward_rec, dsts_pulls_rec, dsts_reward_total, dsts_pulls_total, dsts_full_rec

    # Picking arms, alter probability measure based on how old the info is
    for i in range(startp, startp + n):
        if i >= sw:
            reward_new, pulls_new = sum((i-sw), i, dsts_reward_rec, dsts_pulls_rec)

            rewe_old1 = dsts_reward_total[0] - reward_new[0]
            pulls_old1 = dsts_pulls_total[0] - pulls_new[0]
            rewe_old2 = dsts_reward_total[1] - reward_new[1]
            pulls_old2 = dsts_pulls_total[1] - pulls_new[1]

            arm_1 = np.random.beta(rewe_old1*df + 1,
                                   (pulls_old1 - rewe_old1) * df + 1)
            arm_2 = np.random.beta(rewe_old2*df + 1,
                                   (pulls_old2 - rewe_old2) * df + 1)
        else:
            reward_new, pulls_new = sum(0, i, dsts_reward_rec, dsts_pulls_rec)
            arm_1 = np.random.beta(1, 1)
            arm_2 = np.random.beta(1, 1)

        arm_1_hot = np.random.beta(reward_new[0] + 1,
                                   (pulls_new[0] - reward_new[0]) + 1)
        arm_2_hot = np.random.beta(reward_new[1] + 1,
                                   (pulls_new[1] - reward_new[1]) + 1)

        predicted_prob = [arm_1, arm_2, arm_1_hot, arm_2_hot]
        choice = np.argmax(predicted_prob)

        if choice == 0:
            pull(0, bandit_probs, dsts_reward_total, dsts_pulls_total,
                 dsts_full_rec, 'dsts', dsts_reward_rec, dsts_pulls_rec)
        else:
            pull(1, bandit_probs, dsts_reward_total, dsts_pulls_total, dsts_full_rec, 'dsts')
    return


def pull(arm, bandit_probs, reward_total, pulls_total, full_rec, alg, reward_rec=None, pulls_rec=None):

    if arm == 0:
        a_pull = np.random.choice([0, 1], p=(1 - bandit_probs[0], bandit_probs[0]))
        b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))

        if a_pull == 1 and b_pull == 1:
            reward_total[0] += 1
            pulls_total[0] += 1
            full_rec.append([1, 1])
            if alg == "dsts":
                reward_rec.append([1, 0])
                pulls_rec.append([1, 0])

        elif a_pull == 1 and b_pull == 0:
            reward_total[0] += 1
            pulls_total[0] += 1
            full_rec.append([1, 0])
            if alg == "dsts":
                reward_rec.append([1, 0])
                pulls_rec.append([1, 0])

        elif a_pull == 0 and b_pull == 1:
            pulls_total[0] += 1
            full_rec.append([0, 1])
            if alg == "dsts":
                reward_rec.append([0, 0])
                pulls_rec.append([1, 0])

        else:
            full_rec.append([0, 0])
            pulls_total[0] += 1
            if alg == "dsts":
                reward_rec.append([0, 0])
                pulls_rec.append([1, 0])

    else:
        b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))
        a_pull = np.random.choice([0, 1], p=(1 - bandit_probs[0], bandit_probs[0]))

        if a_pull == 1 and b_pull == 1:
            reward_total[1] += 1
            pulls_total[1] += 1
            full_rec.append([1, 1])
            if alg == 'dsts':
                dsts_reward_rec.append([0, 1])
                dsts_pulls_rec.append([0, 1])

        elif a_pull == 1 and b_pull == 0:
            pulls_total[1] += 1
            full_rec.append([1, 0])
            if alg == 'dsts':
                dsts_reward_rec.append([0, 0])
                dsts_pulls_rec.append([0, 1])

        elif a_pull == 0 and b_pull == 1:
            reward_total[1] += 1
            pulls_total[1] += 1
            full_rec.append([0, 1])
            if alg == 'dsts':
                dsts_reward_rec.append([0, 1])
                dsts_pulls_rec.append([0, 1])

        else:
            full_rec.append([0, 0])
            pulls_total[1] += 1
            if alg == 'dsts':
                dsts_reward_rec.append([0, 0])
                dsts_pulls_rec.append([0, 1])

    return


def sum(start, end, reward, pull):
    reward_sum = [0, 0]
    pull_sum = [0, 0]
    for i in range(start, end):
        reward_sum[0] += reward[i][0]
        pull_sum[0] += pull[i][0]

        reward_sum[1] += reward[i][1]
        pull_sum[1] += pull[i][1]

    return reward_sum, pull_sum


def false_positive (reward, pulls):
    arm1_fp = (pulls[0]-reward[0]) / pulls[0]
    arm2_fp = (pulls[1]-reward[1]) / pulls[1]

    return arm1_fp, arm2_fp


def false_negative(rec, arm):
    total_trials = len(rec)
    num_rewarded = dts_reward_total[arm]
    total_reward = 0
    for subrec in rec:
        total_reward += subrec[arm]

    miss_reward = total_reward - num_rewarded

    return miss_reward/total_trials

