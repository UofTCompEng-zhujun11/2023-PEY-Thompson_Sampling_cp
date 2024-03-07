import numpy as np
from datastructs import dataRecord

#global variable defines(initialize the arm's distribution)
records = dataRecord()


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
    global records

    # Picking arms
    for i in range(startp, startp + n):
        arm_1 = np.random.beta((records.dts_reward_total[0])*df + 1,
                               (records.dts_pulls_total[0] - records.dts_reward_total[0])*df + 1)
        arm_2 = np.random.beta((records.dts_reward_total[1])*df + 1,
                               (records.dts_pulls_total[1] - records.dts_reward_total[1])*df + 1)
        predicted_prob = [arm_1, arm_2]
        choice = np.argmax(predicted_prob)
        records.dts_choice_rec.append(choice)

        # Generate reward based on the choice of arms
        if choice == 0:
            a_pull = np.random.choice([0, 1], p=(1 - bandit_probs[0], bandit_probs[0]))

            if a_pull == 1:
                records.dts_reward_total[0] += 1
                records.dts_pulls_total[0] += 1
            else:
                records.dts_pulls_total[0] += 1

        else:
            b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))

            if b_pull == 1:
                records.dts_reward_total[1] += 1
                records.dts_pulls_total[1] += 1
            else:
                records.dts_pulls_total[1] += 1

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

    global records

    # Picking arms
    for i in range(startp, startp + n):
        arm_1 = np.random.beta((records.dts_reward_total[0])*df + 1,
                               (records.dts_pulls_total[0] - records.dts_reward_total[0])*df + 1)
        arm_2 = np.random.beta((records.dts_reward_total[1])*df + 1,
                               (records.dts_pulls_total[1] - records.dts_reward_total[1])*df + 1)
        predicted_prob = [arm_1, arm_2]
        choice = np.argmax(predicted_prob)
        records.dts_choice_rec.append(choice)

        # Generate reward based on the choice of arms
        if choice == 0:
            pull(0, bandit_probs, records.dts_reward_total, records.dts_pulls_total, records.dts_full_rec, 'dts')
        else:
            pull(1, bandit_probs, records.dts_reward_total, records.dts_pulls_total, records.dts_full_rec, 'dts')

        # if dts_pulls_total[0] != 0 and dts_pulls_total[1] != 0:
        #     dts_false_positive_at_t.append([false_positive(dts_reward_total, dts_pulls_total, 0),
        #                                   false_positive(dts_reward_total, dts_pulls_total)[1]])
        #     dts_power_at_t.append([false_negative(dts_full_rec, 0), false_negative(dts_full_rec, 1)])
        if records.dts_pulls_total[0] != 0 and records.dts_pulls_total[1] == 0:
            records.dts_false_positive_at_t.append([false_positive(records.dts_reward_total, records.dts_pulls_total, 0),
                                             0])
            records.dts_power_at_t.append([false_negative(records.dts_full_rec, 0), 0])

        elif records.dts_pulls_total[0] == 0 and records.dts_pulls_total[1] != 0:
            records.dts_false_positive_at_t.append([0, false_positive(records.dts_reward_total, records.dts_pulls_total, 1)])
            records.dts_power_at_t.append([0, false_negative(records.dts_full_rec, 1)])

        elif records.dsts_pulls_total[0] != 0 and records.dts_pulls_total[1] != 0:
            records.dts_false_positive_at_t.append([false_positive(records.dts_reward_total, records.dts_pulls_total, 0),
                                             false_positive(records.dts_reward_total, records.dts_pulls_total, 1)])
            records.dts_power_at_t.append([false_negative(records.dts_full_rec, 0), false_negative(records.dts_full_rec, 1)])
        else:
            records.dts_false_positive_at_t.append([0, 0])
            records.dts_power_at_t.append([0, 0])

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

    # global dsts_reward_rec, dsts_pulls_rec, dsts_reward_total, dsts_pulls_total
    global records

    # Picking arms, alter probability measure based on how old the info is
    for i in range(startp, startp + n):
        if i >= sw:
            reward_new, pulls_new = sum((i-sw), i, records.dsts_reward_rec, records.dsts_pulls_rec)

            rewe_old1 = records.dsts_reward_total[0] - reward_new[0]
            pulls_old1 = records.dsts_pulls_total[0] - pulls_new[0]
            rewe_old2 = records.dsts_reward_total[1] - reward_new[1]
            pulls_old2 = records.dsts_pulls_total[1] - pulls_new[1]

            arm_1 = np.random.beta(rewe_old1*df + 1,
                                   (pulls_old1 - rewe_old1) * df + 1)
            arm_2 = np.random.beta(rewe_old2*df + 1,
                                   (pulls_old2 - rewe_old2) * df + 1)
        else:
            reward_new, pulls_new = sum(0, i, records.dsts_reward_rec, records.dsts_pulls_rec)
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
                records.dsts_reward_total[0] += 1
                records.dsts_pulls_total[0] += 1
                records.dsts_reward_rec.append([1, 0])
                records.dsts_pulls_rec.append([1, 0])

            else:
                records.dsts_pulls_total[0] += 1
                records.dsts_reward_rec.append([0, 0])
                records.dsts_pulls_rec.append([1, 0])

        else:
            b_pull = np.random.choice([0, 1], p=(1 - bandit_probs[1], bandit_probs[1]))
            if b_pull == 1:
                records.dsts_reward_total[1] += 1
                records.dsts_pulls_total[1] += 1
                records.dsts_reward_rec.append([0, 1])
                records.dsts_pulls_rec.append([0, 1])
            else:
                records.dsts_pulls_total[1] += 1
                records.dsts_reward_rec.append([0, 0])
                records.dsts_pulls_rec.append([0, 1])
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

    # global dsts_reward_rec, dsts_pulls_rec, dsts_reward_total, dsts_pulls_total, \
    #     dsts_full_rec, dsts_false_positive_at_t, dsts_power_at_t
    global records

    # Picking arms, alter probability measure based on how old the info is
    for i in range(startp, startp + n):
        if i >= sw:
            reward_new, pulls_new = sum((i-sw), i, records.dsts_reward_rec, records.dsts_pulls_rec)

            rewe_old1 = records.dsts_reward_total[0] - reward_new[0]
            pulls_old1 = records.dsts_pulls_total[0] - pulls_new[0]
            rewe_old2 = records.dsts_reward_total[1] - reward_new[1]
            pulls_old2 = records.dsts_pulls_total[1] - pulls_new[1]

            arm_1 = np.random.beta(rewe_old1*df + 1,
                                   (pulls_old1 - rewe_old1) * df + 1)
            arm_2 = np.random.beta(rewe_old2*df + 1,
                                   (pulls_old2 - rewe_old2) * df + 1)
        else:
            reward_new, pulls_new = sum(0, i, records.dsts_reward_rec, records.dsts_pulls_rec)
            arm_1 = np.random.beta(1, 1)
            arm_2 = np.random.beta(1, 1)

        arm_1_hot = np.random.beta(reward_new[0] + 1,
                                   (pulls_new[0] - reward_new[0]) + 1)
        arm_2_hot = np.random.beta(reward_new[1] + 1,
                                   (pulls_new[1] - reward_new[1]) + 1)

        predicted_prob = [arm_1, arm_2, arm_1_hot, arm_2_hot]
        choice = np.argmax(predicted_prob)

        if choice == 0:
            pull(0, bandit_probs, records.dsts_reward_total, records.dsts_pulls_total,
                 records.dsts_full_rec, 'dsts', records.dsts_reward_rec, records.dsts_pulls_rec)
        else:
            pull(1, bandit_probs, records.dsts_reward_total, records.dsts_pulls_total, records.dsts_full_rec, 'dsts')

        if records.dsts_pulls_total[0] != 0 and records.dsts_pulls_total[1] == 0:
            records.dsts_false_positive_at_t.append([false_positive(records.dsts_reward_total, records.dsts_pulls_total, 0),
                                            0])
            records.dsts_power_at_t.append([false_negative(records.dsts_full_rec, 0), 0])
        elif records.dsts_pulls_total[0] == 0 and records.dsts_pulls_total[1] != 0:
            records.dsts_false_positive_at_t.append([0, false_positive(records.dsts_reward_total, records.dsts_pulls_total, 1)])
            records.dsts_power_at_t.append([0, false_negative(records.dsts_full_rec, 1)])

        elif records.dsts_pulls_total[0] != 0 and records.dsts_pulls_total[1] != 0:
            records.dsts_false_positive_at_t.append([false_positive(records.dsts_reward_total, records.dsts_pulls_total, 0),
                                             false_positive(records.dsts_reward_total, records.dsts_pulls_total, 1)])
            records.dsts_power_at_t.append([false_negative(records.dsts_full_rec, 0), false_negative(records.dsts_full_rec, 1)])
        else:
            records.dsts_false_positive_at_t.append([0, 0])
            records.dsts_power_at_t.append([0, 0])

            # false_positive(dsts_reward_total, dsts_pulls_total)[1]
            # false_negative(dsts_full_rec, 1)

    return


def pull(arm, bandit_probs, reward_total, pulls_total, full_rec, alg, reward_rec=None, pulls_rec=None):

    global records
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
                records.dsts_reward_rec.append([0, 1])
                records.dsts_pulls_rec.append([0, 1])

        elif a_pull == 1 and b_pull == 0:
            pulls_total[1] += 1
            full_rec.append([1, 0])
            if alg == 'dsts':
                records.dsts_reward_rec.append([0, 0])
                records.dsts_pulls_rec.append([0, 1])

        elif a_pull == 0 and b_pull == 1:
            reward_total[1] += 1
            pulls_total[1] += 1
            full_rec.append([0, 1])
            if alg == 'dsts':
                records.dsts_reward_rec.append([0, 1])
                records.dsts_pulls_rec.append([0, 1])

        else:
            full_rec.append([0, 0])
            pulls_total[1] += 1
            if alg == 'dsts':
                records.dsts_reward_rec.append([0, 0])
                records.dsts_pulls_rec.append([0, 1])

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


def false_positive(reward, pulls, arm):
    arm_fp = (pulls[arm]-reward[arm]) / pulls[arm]
    # arm2_fp = (pulls[1]-reward[1]) / pulls[1]

    return arm_fp


def false_negative(rec, arm):
    global records
    total_trials = len(rec)
    num_rewarded = records.dts_reward_total[arm]
    total_reward = 0
    for subrec in rec:
        total_reward += subrec[arm]

    miss_reward = total_reward - num_rewarded

    return miss_reward/total_trials

