from simfunc import sim_data



def retrieve_reward_at_t(info, arm):
    accumulated_rew = 0
    reward_at_t = []
    for t in info:
        accumulated_rew += t[arm]
        reward_at_t.append(accumulated_rew)

    return reward_at_t


def compile_info (criteria):
    if criteria == 'reward':
        prop_arm_1_dts = retrieve_reward_at_t(records.dts_full_rec, 0)
        prop_arm_1_dsts = retrieve_reward_at_t(records.dsts_full_rec, 0)
    elif criteria == 'power':
        power = sim_data.overall_power
        