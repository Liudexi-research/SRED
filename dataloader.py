import torch
from torch.utils.data import Dataset
import numpy as np
from utils import get_timestamp, get_emotional_fluctuation



class SuicidalDataset(Dataset):
    def __init__(self, label, post_sw, post_nsw, emotion_sw, emotion_nsw, timestamp, liwc, post_hour, current=True, random=False):
        super().__init__()
        self.label = label
        self.post_sw = post_sw
        self.post_nsw = post_nsw
        self.emotion_sw = emotion_sw
        self.emotion_nsw = emotion_nsw
        self.timestamp = timestamp
        self.liwc = liwc
        self.post_hour = post_hour
        self.current = current
        self.random = random

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        labels = torch.tensor(int(self.label[item]))
        if self.current:
            result_sw = self.post_sw[item]
            result_nsw = self.post_nsw[item]
            result_emosw = self.emotion_sw[item]
            result_emonsw = self.emotion_nsw[item]
            if self.random:
                np.random.shuffle(result_sw)
                np.random.shuffle(result_nsw)
                np.random.shuffle(result_emosw)
                np.random.shuffle(result_emonsw)
            sw_features = torch.tensor(result_sw)
            nsw_features = torch.tensor(result_nsw)
            emo_sw_features = torch.tensor(result_emosw)
            emo_nsw_features = torch.tensor(result_emonsw)
            timestamp = torch.tensor(get_timestamp(self.timestamp[item]))
            liwc = torch.tensor(self.liwc[item])
            post_hour = torch.tensor(self.post_hour[item])


        return [labels, sw_features, nsw_features, emo_sw_features, emo_nsw_features, timestamp, liwc, post_hour]
