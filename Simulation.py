import main
from main import discounted_thompson_sampling_for_sim
from main import discounted_sliding_thompson_sampling_for_sim


# Best-arm changes
def simulation_vary_one(arm_prob, df, sim_per_interval, interval, change_by, alg, sw = None):
    sim_result = []
    criteria = []
    for i in range(interval):
        if alg == 'dts':
            discounted_thompson_sampling_for_sim([arm_prob[0], min(arm_prob[1] + change_by*i, 1)],
                                                 sim_per_interval, df,
                                                 (sim_per_interval*i))

            print(f'### Phase {i+1} results ###')

            arm1_mean = main.dts_reward_total[0]/main.dts_pulls_total[0]
            arm2_mean = main.dts_reward_total[1]/main.dts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            criteria.append([main.false_positive(main.dts_reward_total,
                                                 main.dts_pulls_total),
                             main.false_negative(main.dts_full_rec, 0),
                             main.false_negative(main.dts_full_rec, 1),
                             main.dts_reward_total[0], main.dts_reward_total[1]])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')

            print(f' Arm 1 Reward = {main.dts_reward_total[0]},'
                  f' Arm 2 Reward = {main.dts_reward_total[1]} ')

            print(f' Arm 1 Power = {main.false_negative(main.dts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dts_full_rec, 1)} ')

            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[1]} ')

        else:
            discounted_sliding_thompson_sampling_for_sim([arm_prob[0], min(arm_prob[1] + change_by*i, 1)],
                                                         sim_per_interval, df, sw,
                                                         (sim_per_interval*i))
            print(f'### Phase {i+1} results ###')

            arm1_mean = main.dsts_reward_total[0]/main.dsts_pulls_total[0]
            arm2_mean = main.dsts_reward_total[1]/main.dsts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            criteria.append([main.false_positive(main.dsts_reward_total,
                                                 main.dsts_pulls_total),
                             main.false_negative(main.dsts_full_rec, 0),
                             main.false_negative(main.dsts_full_rec, 1),
                             main.dts_reward_total[0], main.dts_reward_total[1]])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')

            print(f' Arm 1 Reward = {main.dsts_reward_total[0]},'
                  f' Arm 2 Reward = {main.dsts_reward_total[1]} ')

            print(f' Arm 1 Power = {main.false_negative(main.dsts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dsts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[1]} ')

    return sim_result, criteria


# Best-arm changes slowly,(both varies)
def simulation_vary_both(arm_prob, df, sim_per_interval, interval, change_by, alg, sw=None):
    sim_result = []
    criteria = []
    for i in range(interval):
        if alg == 'dts':
            discounted_thompson_sampling_for_sim([min(arm_prob[0] + change_by*i, 1),
                                                 min(arm_prob[1] + change_by*i, 1)],
                                                 sim_per_interval, df,
                                                 (sim_per_interval*i))
            print(f'### Phase {i} results ###')

            arm1_mean = main.dts_reward_total[0]/main.dts_pulls_total[0]
            arm2_mean = main.dts_reward_total[1]/main.dts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            criteria.append([main.false_positive(main.dts_reward_total,
                                                 main.dts_pulls_total),
                             main.false_negative(main.dts_full_rec, 0),
                             main.false_negative(main.dts_full_rec, 1),
                             main.dts_reward_total[0], main.dts_reward_total[1]])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')

            print(f' Arm 1 Reward = {main.dts_reward_total[0]},'
                  f' Arm 2 Reward = {main.dts_reward_total[1]} ')

            print(f' Arm 1 Power = {main.false_negative(main.dts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[1]} ')

        else:
            discounted_sliding_thompson_sampling_for_sim([min(arm_prob[0] + change_by*i, 1),
                                                          min(arm_prob[1] + change_by*i, 1)],
                                                         sim_per_interval, df, sw,
                                                         (sim_per_interval*i))
            print(f'### Phase {i+1} results ###')

            arm1_mean = main.dsts_reward_total[0]/main.dsts_pulls_total[0]
            arm2_mean = main.dsts_reward_total[1]/main.dsts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')
            criteria.append([main.false_positive(main.dsts_reward_total,
                                                 main.dsts_pulls_total),
                             main.false_negative(main.dsts_full_rec, 0),
                             main.false_negative(main.dsts_full_rec, 1),
                             main.dsts_reward_total[0], main.dsts_reward_total[1]])

            print(f' Arm 1 Reward = {main.dsts_reward_total[0]},'
                  f' Arm 2 Reward = {main.dsts_reward_total[1]} ')

            print(f' Arm 1 Power = {main.false_negative(main.dsts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dsts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[1]} ')

    return sim_result, criteria


def simulation_vary_both_random(arm_prob, df, sim_per_interval, interval, func, alg, sw=None):
    sim_result = []
    criteria = []
    for i in range(interval):
        if alg == 'dts':
            discounted_thompson_sampling_for_sim([min(arm_prob[0] + func*i, 1),
                                                  min(arm_prob[1] + func*i, 1)],
                                                 sim_per_interval, df,
                                                 (sim_per_interval*i))
            print(f'### Phase {i} results ###')

            arm1_mean = main.dts_reward_total[0]/main.dts_pulls_total[0]
            arm2_mean = main.dts_reward_total[1]/main.dts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')
            criteria.append([main.false_positive(main.dts_reward_total,
                                                 main.dts_pulls_total),
                             main.false_negative(main.dts_full_rec, 0),
                             main.false_negative(main.dts_full_rec, 1)])

            print(f' Arm 1 Power = {main.false_negative(main.dts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[1]} ')

        else:
            discounted_sliding_thompson_sampling_for_sim([min(arm_prob[0] + func*i, 1),
                                                          min(arm_prob[1] + func*i, 1)],
                                                         sim_per_interval, df, sw,
                                                         (sim_per_interval*i))
            print(f'### Phase {i+1} results ###')

            arm1_mean = main.dsts_reward_total[0]/main.dsts_pulls_total[0]
            arm2_mean = main.dsts_reward_total[1]/main.dsts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')
            criteria.append([main.false_positive(main.dsts_reward_total,
                                                 main.dsts_pulls_total),
                             main.false_negative(main.dsts_full_rec, 0),
                             main.false_negative(main.dsts_full_rec, 1)])

            print(f' Arm 1 Power = {main.false_negative(main.dsts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dsts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[1]} ')

    return sim_result, criteria


# Best-arm changes slowly,(both varies)
def simulation_vary_both(arm_prob, df, sim_per_interval, interval, change_by, alg, sw=None):
    sim_result = []
    criteria = []
    for i in range(interval):
        if alg == 'dts':
            discounted_thompson_sampling_for_sim([min(arm_prob[0] + change_by*i, 1),
                                                 min(arm_prob[1] + change_by*i, 1)],
                                                 sim_per_interval, df,
                                                 (sim_per_interval*i))
            print(f'### Phase {i} results ###')

            arm1_mean = main.dts_reward_total[0]/main.dts_pulls_total[0]
            arm2_mean = main.dts_reward_total[1]/main.dts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')
            criteria.append([main.false_positive(main.dts_reward_total,
                                                 main.dts_pulls_total),
                             main.false_negative(main.dts_full_rec, 0),
                             main.false_negative(main.dts_full_rec, 1)])

            print(f' Arm 1 Power = {main.false_negative(main.dts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dts_reward_total, main.dts_pulls_total)[1]} ')

        else:
            discounted_sliding_thompson_sampling_for_sim([min(arm_prob[0] + change_by*i, 1),
                                                          min(arm_prob[1] + change_by*i, 1)],
                                                         sim_per_interval, df, sw,
                                                         (sim_per_interval*i))
            print(f'### Phase {i+1} results ###')

            arm1_mean = main.dsts_reward_total[0]/main.dsts_pulls_total[0]
            arm2_mean = main.dsts_reward_total[1]/main.dsts_pulls_total[1]
            sim_result.append([arm1_mean, arm2_mean])

            print(f' Arm 1 Mean = {arm1_mean}, Arm 2 Mean = {arm2_mean}')
            criteria.append([main.false_positive(main.dsts_reward_total,
                                                 main.dsts_pulls_total),
                             main.false_negative(main.dsts_full_rec, 0),
                             main.false_negative(main.dsts_full_rec, 1)])

            print(f' Arm 1 Power = {main.false_negative(main.dsts_full_rec, 0)},'
                  f' Arm 2 Power = {main.false_negative(main.dsts_full_rec, 1)} ')
            print(f' Arm 1 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[0]},'
                  f' Arm 2 False Positive Rates = {main.false_positive(main.dsts_reward_total, main.dsts_pulls_total)[1]} ')

    return sim_result, criteria


simulation_vary_one([0.3, 0.4], 0.8, 500, 3, 0.02, 'dts')

simulation_vary_one([0.2, 0.3], 0.8, 500, 5, 0.02, 'dsts', 5)


