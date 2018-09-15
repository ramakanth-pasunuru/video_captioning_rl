import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F



class biDafAttn(nn.Module):
    def __init__(self, channel_size):
        super(biDafAttn, self).__init__()
        """
        This method do biDaf from s2 to s1:
            The return value will have the same size as s1.
        :param channel_size: Hidden size of the input
        """
        # self.mlp = nn.Linear(channel_size * 3, 1, bias=False)

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        batch_size = s1.size(0)
        t1 = s1.size(1)
        t2 = s2.size(1)
        S = torch.bmm(s1, s2.transpose(1,
                                       2))  # [B, t1, D] * [B, D, t2] -> [B, t1, t2] S is the similarity matrix from biDAF paper. [B, T1, T2]

        s_mask = S.data.new(*S.size()).fill_(1).byte()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        s_mask = Variable(s_mask)
        S.data.masked_fill_(s_mask.data.byte(), -math.inf)
        return S

    def get_U_tile(self, S, s2):
        a_weight = F.softmax(S, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        U_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        return U_tile

    def get_both_tile(self, S, s1, s2):
        a_weight = F.softmax(S, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        U_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]

        a1_weight = F.softmax(S, dim=1)  # [B, t1, t2]
        a1_weight.data.masked_fill_(a1_weight.data != a1_weight.data, 0)  # remove nan from softmax on -inf
        U1_tile = torch.bmm(a1_weight.transpose(1, 2), s1)  # [B, t2, t1] * [B, t1, D] -> [B, t2, D]
        return U_tile, U1_tile

    def forward(self, s1, l1, s2, l2):
        S = self.similarity(s1, l1, s2, l2)
        U_tile = self.get_U_tile(S, s2)
        return U_tile


class CoattMaxPool(nn.Module):
    def __init__(self, args):
         
        super(CoattMaxPool, self).__init__()
        h_size = [300, 300]
        d = 300
        mlp_d = 300
        v_size=args.max_snli_vocab_size
        max_l= None
        num_of_class=3
        drop_r=args.dropout
        featurizer=None
        itos=None
        with_emlo=False
        activation_type='relu'
        self.h_size = h_size
        self.e_embd = nn.Embedding(v_size, d)
        self.embd_dropout = nn.Dropout(drop_r)
        self.featurizer = featurizer
        self.itos = itos
        self.args = args
        if self.featurizer is not None:
            fcount = self.featurizer.n_context_features()
        else:
            fcount = 0

        self.emlo_embedding_d = 0
        if with_emlo:
            self.emlo_ee = ElmoEmbedder(cuda_device=n_device)
            self.emlo_embedding_d = 1024

        self.emlo_gamma = nn.Parameter(torch.FloatTensor([1]))
        self.emlo_s_vector = nn.Parameter(torch.FloatTensor([1, 1, 1]))



        if self.args.cell_type=='gru':
            self.lstm = nn.GRU(input_size=d + fcount + self.emlo_embedding_d * 1, hidden_size=h_size[0],
                        num_layers=1, bidirectional=True, batch_first=True)

            self.lstm_1 = nn.GRU(input_size=h_size[1] + fcount + self.emlo_embedding_d * 1, hidden_size=h_size[1],
                          num_layers=1, bidirectional=True, batch_first=True)

        else:
            self.lstm = nn.LSTM(input_size=d + fcount + self.emlo_embedding_d * 1, hidden_size=h_size[0],
                        num_layers=1, bidirectional=True, batch_first=True)

            self.lstm_1 = nn.LSTM(input_size=h_size[1] + fcount + self.emlo_embedding_d * 1, hidden_size=h_size[1],
                          num_layers=1, bidirectional=True, batch_first=True)

        self.projection = nn.Linear(h_size[0] * 2 * 4, h_size[1])
        self.projection_dropout = nn.Dropout(drop_r)

        self.max_l = max_l
        self.bidaf = biDafAttn(300)

        self.mlp_1 = nn.Linear(h_size[1] * 2 * 4, mlp_d)
        self.sm = nn.Linear(mlp_d, num_of_class)
        if activation_type == 'relu':
            activation = nn.ReLU()
            # self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(drop_r), self.sm])
        elif activation_type == 'tanh':
            activation = nn.Tanh()
        else:
            raise ValueError("Not a valid activation!")

        self.classifier = nn.Sequential(*[nn.Dropout(drop_r), self.mlp_1, activation, nn.Dropout(drop_r), self.sm])





    def count_params(self):
        total_c = 0
        for param in self.parameters():
            if len(param.size()) == 2:
                d1, d2 = param.size()[0], param.size()[1]
                total_c += d1 * d2
        print("Total count:", total_c)

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def forward(self, s1, l1, s2, l2):  # [B, T]
        if self.max_l:
            max_l = min(s1.size(1), self.max_l)
            max_l = max(1, max_l)
            max_s1_l = min(max(l1), max_l)

            l1 = l1.clamp(min=1, max=max_s1_l)
            if s1.size(1) > max_s1_l:
                s1 = s1[:, :max_s1_l]

        s1_max_l = s1.size(1)

        if self.max_l:
            max_l = min(s2.size(1), self.max_l)
            max_l = max(1, max_l)
            max_s2_l = min(max(l2), max_l)

            l2 = l2.clamp(min=1, max=max_s2_l)
            if s2.size(1) > max_s2_l:
                s2 = s2[:, :max_s2_l]

        s2_max_l = s2.size(1)

        batch_size = s1.size(0)

        th_packed_f_s1, th_packed_f_s2 = None, None
        emlo_s1_sum, emlo_s2_sum = None, None



        p_s1 = self.e_embd(s1)
        p_s2 = self.e_embd(s2)

        p_s1 = self.embd_dropout(p_s1)  # Embedding dropout
        p_s2 = self.embd_dropout(p_s2)  # Embedding dropout

        feature_p_s1 = torch.cat([seq for seq in [p_s1, th_packed_f_s1, emlo_s1_sum] if seq is not None], dim=2)
        feature_p_s2 = torch.cat([seq for seq in [p_s2, th_packed_f_s2, emlo_s2_sum] if seq is not None], dim=2)

        s1_layer1_out = self.auto_rnn(self.lstm, feature_p_s1, l1)
        s2_layer1_out = self.auto_rnn(self.lstm, feature_p_s2, l2)

        S = self.bidaf.similarity(s1_layer1_out, l1, s2_layer1_out, l2)
        s1_att, s2_att = self.bidaf.get_both_tile(S, s1_layer1_out, s2_layer1_out)



        s1_coattentioned = torch.cat([s1_layer1_out, s1_att, s1_layer1_out - s1_att,
                                      s1_layer1_out * s1_att], dim=2)

        s2_coattentioned = torch.cat([s2_layer1_out, s2_att, s2_layer1_out - s2_att,
                                      s2_layer1_out * s2_att], dim=2)

        p_s1_coattentioned = self.projection_dropout(F.relu(self.projection(s1_coattentioned)))
        p_s2_coattentioned = self.projection_dropout(F.relu(self.projection(s2_coattentioned)))

        s1_coatt_features = torch.cat(
            [seq for seq in [p_s1_coattentioned, th_packed_f_s1, emlo_s1_sum] if seq is not None], dim=2)
        s2_coatt_features = torch.cat(
            [seq for seq in [p_s2_coattentioned, th_packed_f_s2, emlo_s2_sum] if seq is not None], dim=2)

        s1_layer2_out = self.auto_rnn(self.lstm_1, s1_coatt_features, l1)
        s2_layer2_out = self.auto_rnn(self.lstm_1, s2_coatt_features, l2)

        s1_lay2_maxout = max_along_time(s1_layer2_out, l1)
        s2_lay2_maxout = max_along_time(s2_layer2_out, l2)

        s1_lay2_avgout = avg_along_time(s1_layer2_out, l1)
        s2_lay2_avgout = avg_along_time(s2_layer2_out, l2)

        features = torch.cat([s1_lay2_maxout, s2_lay2_maxout,
                              s1_lay2_avgout, s2_lay2_avgout], dim=1)

        logits = self.classifier(features)
        probs = F.softmax(logits, 1)
        pred = torch.max(probs, 1)[1]

        return logits, probs, pred


    
    def auto_rnn(self, rnn: nn.RNN, seqs, lengths, batch_first=True, init_state=None, output_last_states=False):
        batch_size = seqs.size(0) if batch_first else seqs.size(1)
        state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

        if not init_state:
            h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
        else:
            h0 = init_state['h0'].expand(state_shape)
            c0 = init_state['c0'].expand(state_shape)

        packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths, batch_first)
        if self.args.cell_type == 'gru':
            output, hn = rnn(packed_pinputs, h0)
        else:
            output, (hn, cn) = rnn(packed_pinputs, (h0, c0))
        output = unpack_from_rnn_seq(output, r_index, batch_first)

        if not output_last_states:
            return output
        else:
            return output, (hn, cn)




def pad_1d(seq, pad_l):
    """
    The seq is a sequence having shape [T, ..]. Note: The seq contains only one instance. This is not batched.
    
    :param seq:  Input sequence with shape [T, ...]
    :param pad_l: The required pad_length.
    :return:  Output sequence will have shape [Pad_L, ...]
    """
    l = seq.size(0)
    if l >= pad_l:
        return seq[:pad_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = Variable(seq.data.new(pad_l - l, *seq.size()[1:]).zero_())  # Requires_grad is False
        return torch.cat([seq, pad_seq], dim=0)


def pad(seqs, length, batch_first=True):
    #TODO The method seems useless to me. Delete?
    """
    Padding the sequence to a fixed length.
    
    :param seqs: [B, T, D] or [B, T] if batch_first else [T, B * D] or [T, B]
    :param length: [B]
    :param batch_first:
    :return:
    """
    if batch_first:
        # [B * T * D]
        if length <= seqs.size(1):
            return seqs[:, :length]
        else:
            batch_size = seqs.size(0)
            pad_seq = Variable(seqs.data.new(batch_size, length - seqs.size(1), *seqs.size()[2:]).zero_())
            # [B * T * D]
            return torch.cat([seqs, pad_seq], dim=1)
    else:
        # [T * B * D]
        if length <= seqs.size(0):
            return seqs
        else:
            return torch.cat([seqs, Variable(seqs.data.new(length - seqs.size(0), *seqs.size()[1:]).zero_())])


def batch_first2time_first(inputs):
    """
    Convert input from batch_first to time_first:
    [B, T, D] -> [T, B, D]
    
    :param inputs:
    :return:
    """
    return torch.transpose(inputs, 0, 1)


def time_first2batch_first(inputs):
    """
    Convert input from batch_first to time_first:
    [T, B, D] -> [B, T, D] 
    
    :param inputs:
    :return:
    """
    return torch.transpose(inputs, 0, 1)


def get_state_shape(rnn: nn.RNN, batch_size, bidirectional=False):
    """
    Return the state shape of a given RNN. This is helpful when you want to create a init state for RNN.
    Example:
    c0 = h0 = Variable(src_seq_p.data.new(*get_state_shape([your rnn], 3, bidirectional)).zero_())
    
    :param rnn: nn.LSTM, nn.GRU or subclass of nn.RNN
    :param batch_size:  
    :param bidirectional:  
    :return: 
    """
    if bidirectional:
        return rnn.num_layers * 2, batch_size, rnn.hidden_size
    else:
        return rnn.num_layers, batch_size, rnn.hidden_size


def pack_list_sequence(inputs, l, max_l=None, batch_first=True):
    """
    Pack a batch of Tensor into one Tensor with max_length.
    :param inputs: 
    :param l: 
    :param max_l: The max_length of the packed sequence.
    :param batch_first: 
    :return: 
    """
    batch_list = []
    max_l = max(list(l)) if not max_l else max_l
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad_1d(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1) if not batch_first \
        else torch.stack(batch_list, dim=0)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths, batch_first=True):
    """
    :param inputs: Shape of the input should be [B, T, D] if batch_first else [T, B, D].
    :param lengths:  [B]
    :param batch_first: 
    :return: 
    """
    if not batch_first:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

        return packed_seq, reverse_indices

    else:
        #print(lengths)
        #_, sorted_indices = lengths.sort()
        r_index = reversed(list(np.argsort(lengths)))
        '''
            Reverse to decreasing order
        '''
        #r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        #reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)
        reverse_indices = np.zeros(len(lengths), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[i, :, :])
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.stack(s_inputs_list, dim=0)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list, batch_first=batch_first)

        return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices, batch_first=True):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=batch_first)
    s_inputs_list = []

    if not batch_first:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)
    else:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[i, :, :].unsqueeze(0))
        return torch.cat(s_inputs_list, 0)


def auto_rnn(rnn: nn.RNN, seqs, lengths, batch_first=True, init_state=None, output_last_states=False):
    batch_size = seqs.size(0) if batch_first else seqs.size(1)
    state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

    if not init_state:
        h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
    else:
        h0 = init_state['h0'].expand(state_shape)
        c0 = init_state['c0'].expand(state_shape)

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths, batch_first)
    output, (hn, cn) = rnn(packed_pinputs, (h0, c0))
    output = unpack_from_rnn_seq(output, r_index, batch_first)

    if not output_last_states:
        return output
    else:
        return output, (hn, cn)


def pack_sequence_for_linear(inputs, lengths, batch_first=True):
    """
    :param inputs: [B, T, D] if batch_first 
    :param lengths:  [B]
    :param batch_first:  
    :return: 
    """
    batch_list = []
    if batch_first:
        for i, l in enumerate(lengths):
            # print(inputs[i, :l].size())
            batch_list.append(inputs[i, :l])
        packed_sequence = torch.cat(batch_list, 0)
        # if chuck:
        #     return list(torch.chunk(packed_sequence, chuck, dim=0))
        # else:
        return packed_sequence
    else:
        raise NotImplemented()


def chucked_forward(inputs, net, chuck=None):
    if not chuck:
        return net(inputs)
    else:
        output_list = [net(chuck) for chuck in torch.chunk(inputs, chuck, dim=0)]
        return torch.cat(output_list, dim=0)


def unpack_sequence_for_linear(inputs, lengths, batch_first=True):
    batch_list = []
    max_l = max(lengths)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    if batch_first:
        start = 0
        for l in lengths:
            end = start + l
            batch_list.append(pad_1d(inputs[start:end], max_l))
            start = end
        return torch.stack(batch_list)
    else:
        raise NotImplemented()


def seq2seq_cross_entropy(logits, label, l, chuck=None, sos_truncate=True):
    """
    :param logits: [exB, V] : exB = sum(l)
    :param label: [B] : a batch of Label
    :param l: [B] : a batch of LongTensor indicating the lengths of each inputs
    :param chuck: Number of chuck to process
    :return: A loss value
    """
    packed_label = pack_sequence_for_linear(label, l)
    cross_entropy_loss = functools.partial(F.cross_entropy, size_average=False)
    total = sum(l)

    assert total == logits.size(0) or packed_label.size(0) == logits.size(0),\
        "logits length mismatch with label length."

    if chuck:
        logits_losses = 0
        for x, y in zip(torch.chunk(logits, chuck, dim=0), torch.chunk(packed_label, chuck, dim=0)):
            logits_losses += cross_entropy_loss(x, y)
        return logits_losses * (1 / total)
    else:
        return cross_entropy_loss(logits, packed_label) * (1 / total)


def avg_along_time(inputs, lengths, list_in=False):
    """
    :param inputs: [B, T, D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    :param list_in: 
    """
    ls = list(lengths)

    if not list_in:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_max = seq_i.mean(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)
    else:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i]
            seq_i_max = seq_i.mean(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)



def max_along_time(inputs, lengths, list_in=False):
    """
    :param inputs: [B, T, D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    :param list_in: 
    """
    ls = list(lengths)

    if not list_in:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)
    else:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)



