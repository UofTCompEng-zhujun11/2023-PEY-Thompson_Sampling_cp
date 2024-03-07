class dataRecord:

    def __init__(self) -> None:
        self.dsts_reward_rec = list(list())
        self.dsts_pulls_rec = list(list())
        self.dsts_reward_total = [0, 0]
        self.dsts_pulls_total = [0, 0]
    
        self.dsts_false_positive_at_t = []
        self.dsts_power_at_t = []
        self.dsts_full_rec = list(list())
    
        self.dts_choice_rec = list()
        self.dts_reward_total = [0, 0]
        self.dts_pulls_total = [0, 0]
    
        self.dts_false_positive_at_t = []
        self.dts_power_at_t = []
        self.dts_full_rec = list(list())

    def clearData(self) -> None:
        self.dsts_reward_rec.clear()
        self.dsts_pulls_rec.clear()
        self.dsts_reward_total = [0, 0]
        self.dsts_pulls_total = [0, 0]
    
        self.dsts_false_positive_at_t.clear()
        self.dsts_power_at_t.clear()
        self.dsts_full_rec.clear()
    
        self.dts_choice_rec.clear()
        self.dts_reward_total = [0, 0]
        self.dts_pulls_total = [0, 0]
    
        self.dts_false_positive_at_t.clear()
        self.dts_power_at_t.clear()
        self.dts_full_rec.clear()
