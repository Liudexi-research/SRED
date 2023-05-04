import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn



def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def pad_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens]


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    sw = [item[1] for item in batch]
    nsw = [item[2] for item in batch]
    emo_sw = [item[3] for item in batch]
    emo_nsw = [item[4] for item in batch]
    timestamp = [item[5] for item in batch]
    liwc = [item[6] for item in batch]
    post_hour = [item[7] for item in batch]
    sw_lens = [len(x) for x in sw]
    nsw_lens = [len(x) for x in nsw]

    sw = nn.utils.rnn.pad_sequence(sw, batch_first=True, padding_value=0)
    nsw = nn.utils.rnn.pad_sequence(nsw, batch_first=True, padding_value=0)
    emo_sw = nn.utils.rnn.pad_sequence(emo_sw, batch_first=True, padding_value=0)
    emo_nsw = nn.utils.rnn.pad_sequence(emo_nsw, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)
    liwc = nn.utils.rnn.pad_sequence(liwc, batch_first=True, padding_value=0)
    post_hour = nn.utils.rnn.pad_sequence(post_hour, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    sw_lens = torch.tensor(sw_lens)
    nsw_lens = torch.tensor(nsw_lens)
    return [target, sw, nsw, emo_sw, emo_nsw, timestamp, sw_lens, nsw_lens, liwc, post_hour]


def get_emotional_fluctuation(x):
    embedding = x.tolist()
    fluctuation = []
    fluctuation.append(embedding[0])
    for i in range(len(embedding)-1):
        fluc = list(map(lambda x: x[0]-x[1], zip(embedding[i+1],embedding[i])))
        fluctuation.append(fluc)
    fluctuation = np.array(fluctuation, dtype='float32')
    return fluctuation


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    df = pd.DataFrame({'timestamp':timestamp})
    interval = df['timestamp'].diff()
    interval.fillna(1, inplace=True)
    interval = [1/(t+1) for t in interval]
    timestamp = np.array(interval, dtype='float32')

    return timestamp
