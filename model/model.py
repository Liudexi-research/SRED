from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from torch import nn


class BiLSTMAttn(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim // 2, dropout=dropout if num_layers > 1 else 0,
                               num_layers=num_layers, batch_first=True, bidirectional=True)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden

    def forward(self, features, lens):
        features = self.dropout(features)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True, enforce_sorted=False)
        outputs, (hn, cn) = self.encoder(packed_embedded)
        outputs, output_len = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        fbout = outputs[:, :, :self.hidden_dim // 2] + outputs[:, :, self.hidden_dim // 2:]
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        attn_out = self.attnetwork(fbout, fbhn)

        return attn_out  #batch_size, hidden_dim/2


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, dropout=dropout, num_layers=num_layers, batch_first=True,
                              bidirectional=True)

    def forward(self, features, lens):
        # print(self.hidden.size())
        features = self.dropout(features)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True, enforce_sorted=False)
        outputs, hidden_state = self.bilstm(packed_embedded)
        outputs, output_len = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden_state  # outputs: batch, seq, hidden_dim - hidden_state: hn, cn: 2*num_layer, batch_size, hidden_dim/2


class DualContext(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, model):
        super().__init__()
        self.model = model
        if self.model == "tlstm":
            self.historic_model = TimeLSTM(embedding_dim, hidden_dim)
        elif self.model == "bilstm":
            self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        elif self.model == "bilstm-attention":
            self.historic_model = BiLSTMAttn(embedding_dim, hidden_dim, num_layers, dropout)

        self.sw_model = BiLSTMAttn(embedding_dim, hidden_dim, num_layers, dropout)
        self.full_concat = nn.Linear(768, 384)
        self.full_concat.to("cuda")
        self.fc_ct = nn.Linear(768, hidden_dim)
        self.fc_ct_attn = nn.Linear(768, hidden_dim // 2)

        # self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_concat = nn.Linear(368, hidden_dim)
        self.fc_concat_attn = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        # self.final = nn.Linear(hidden_dim, 2)
        self.final = nn.Linear(hidden_dim, 4)

    @staticmethod
    def combine_features(tweet_features, historic_features):
        return torch.cat((tweet_features, historic_features), 1)

    @staticmethod
    def bert_combine_emotion(tweet_features, historic_features):
        return torch.cat((tweet_features, historic_features), 2)

    def forward(self, sw_features, nsw_features, emotion_sw, emotion_nsw, sw_lens, nsw_lens, timestamp, liwc,
                post_hour):
        if self.model == "tlstm":

            # bert_embedding 和emotion_embedding拼接
            sw_features = self.full_concat(sw_features)
            emotion_sw = self.full_concat(emotion_sw)
            sw_features = self.bert_combine_emotion(sw_features, emotion_sw)

            nsw_features = self.full_concat(nsw_features)
            emotion_nsw = self.full_concat(emotion_nsw)
            nsw_features = self.bert_combine_emotion(nsw_features, emotion_nsw)

            sw_outputs = self.sw_model(sw_features, sw_lens.to('cpu'))
            nsw_outputs = self.historic_model(nsw_features, timestamp)
            combined_features = self.combine_features(sw_outputs, nsw_outputs)
            liwc = F.softmax(liwc, dim=1)
            user_features = self.combine_features(combined_features, liwc)
            user_features = self.combine_features(user_features, post_hour)
            user_features = self.dropout(user_features)
            x = F.relu(self.fc_concat(user_features))


        elif self.model == "bilstm":

            sw_outputs, (h_n, c_n) = self.historic_model(sw_features, sw_lens)
            sw_outputs = torch.mean(sw_outputs, 1)
            nsw_outputs, (h_n, c_n) = self.historic_model(nsw_features, nsw_lens)
            nsw_outputs = torch.mean(nsw_outputs, 1)
            combined_features = self.combine_features(sw_outputs, nsw_outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat(combined_features))


        elif self.model == "bilstm-attention":
            # bert_embedding 和emotion_embedding拼接

            sw_features = self.full_concat(sw_features)
            emotion_sw = self.full_concat(emotion_sw)
            sw_features = self.bert_combine_emotion(sw_features, emotion_sw)

            nsw_features = self.full_concat(nsw_features)
            emotion_nsw = self.full_concat(emotion_nsw)
            nsw_features = self.bert_combine_emotion(nsw_features, emotion_nsw)


            sw_outputs = self.historic_model(sw_features, sw_lens.to('cpu'))
            nsw_outputs = self.historic_model(nsw_features, nsw_lens.to('cpu'))
            combined_features = self.combine_features(sw_outputs, nsw_outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat_attn(combined_features))

        x = self.dropout(x)

        return self.final(x)


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        # assumes that batch_first is always true
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        score_x = x * att_score
        context = torch.sum(score_x, dim=1)
        return context

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)

        h = h.cuda()
        c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        attn_outputs = self.attention_net(outputs)
        return attn_outputs
