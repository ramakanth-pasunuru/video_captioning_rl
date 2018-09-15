import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import numpy as np
import bottleneck as bn

def rnn_mask(context_lens, max_step):
    """
    Creates a mask for variable length sequences
    """
    num_batches = len(context_lens)

    mask = torch.FloatTensor(num_batches, max_step).zero_()
    if torch.cuda.is_available():
        mask = mask.cuda()
    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1.0
    mask = Variable(mask)
    return mask

def top_n_indexes(arr, n):
        idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
        width = arr.shape[1]
        return [divmod(i, width) for i in idx]

class Seq2seqAttention(nn.Module):
    def __init__(self, args):
        super(Seq2seqAttention, self).__init__()
        self.args = args
        self.enable_cuda = args.cuda
        self.vid_dim = args.vid_dim
        self.embed_size = args.embed
        self.hidden_dim = args.hid
        self.vocab_size = args.max_vocab_size
        self.num_layers = args.num_layers
        self.birnn = args.birnn
        self.encoder = EncoderFrames(self.args)
        self.decoder = DecoderRNN(self.args)

    def forward(self, frames, flengths, captions, lengths):
        video_features = self.encoder(frames, flengths)
        outputs = self.decoder(video_features, flengths, captions, lengths)

        return outputs



    def sample(self, frames, flengths):
        video_features = self.encoder.forward(frames, flengths)
        predicted_target = self.decoder.sample(video_features, flengths)
        return predicted_target

    def sample_rl(self, frames, flengths, sampling='multinomial'):
        video_features = self.encoder.forward(frames, flengths)
        predicted_target, outputs = self.decoder.rl_sample(video_features, flengths, sampling=sampling)
        return predicted_target, outputs

    def beam_search(self, frames, flengths, beam_size=5):
        video_features = self.encoder.forward(frames, flengths)
        predicted_target = self.decoder.beam_search(video_features, flengths, beam_size=beam_size)

        return predicted_target






# Based on tutorials/08 - Language Model
# RNN Based Language Model
class EncoderFrames(nn.Module):
    def __init__(self, args):
        super(EncoderFrames, self).__init__()
        # self.use_abs = use_abs
        self.vid_dim = args.vid_dim
        self.embed_size = args.embed
        self.hidden_dim = args.hid
        self.enable_cuda = args.cuda
        self.num_layers = args.num_layers
        self.args = args
        if args.birnn:
            self.birnn = 2
        else:
            self.birnn = 1
        # projection layer
        self.linear = nn.Linear(self.vid_dim, self.embed_size, bias=False)
        # video embedding
        self.rnn = nn.LSTM(self.embed_size, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.args.birnn, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.init_weights()

    def init_weights(self):
        self.rnn.weight_hh_l0.data.uniform_(-0.08, 0.08)
        self.rnn.weight_ih_l0.data.uniform_(-0.08, 0.08)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.linear.weight.data.uniform_(-0.08, 0.08)
        #self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        if self.birnn:
            return (Variable(torch.zeros(self.birnn*self.num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(self.birnn*self.num_layers, batch_size, self.hidden_dim)))



    def forward(self, frames, flengths):
        """Handles variable size frames
           frame_embed: video features
           flengths: frame lengths
        """
        batch_size = flengths.shape[0]
        #frames = self.linear(frames)
        #frames = self.dropout(frames) # adding dropout layer
        self.init_rnn = self.init_hidden(batch_size)
        if self.enable_cuda:
            self.init_rnn = self.init_rnn[0].cuda(), self.init_rnn[1].cuda()

        if batch_size > 1:
            # Sort by length (keep idx)
            flengths, idx_sort = np.sort(flengths)[::-1], np.argsort(-flengths)
            if self.enable_cuda:
                frames = frames.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            else:
                frames = frames.index_select(0, Variable(torch.LongTensor(idx_sort)))



        frames = self.linear(frames)
        frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths, batch_first=True)
        outputs, (ht, ct) = self.rnn(frame_packed, self.init_rnn)
        outputs,_ = pad_packed_sequence(outputs,batch_first=True)

        if batch_size > 1:
            # Un-sort by length
            idx_unsort = np.argsort(idx_sort)
            if self.enable_cuda:
                outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))
            else:
                outputs = outputs.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        # print 'Encoder Outputs:',outputs.size()

        return outputs


# Based on tutorials/03 - Image Captioning
class DecoderRNN(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.enable_cuda = args.cuda
        self.embed_size = args.embed
        self.hidden_size = args.hid
        self.vocab_size = args.max_vocab_size
        if args.birnn:
            self.birnn = 2
        else:
            self.birnn = 1

        self.num_layers = args.num_layers
        self.args = args
        self.input_proj = nn.Linear(self.birnn*self.hidden_size+self.embed_size, self.embed_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.atten = Attention(args, self.birnn*self.hidden_size, self.hidden_size)  
        self.lstm = nn.LSTM(self.embed_size+self.birnn*self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=args.dropout)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        #self.lstm.weight_hh_l0.data.uniform_(-0.08, 0.08)
        self.lstm.weight_hh_l0.data.uniform_(-0.08, 0.08)
        self.lstm.weight_ih_l0.data.uniform_(-0.08, 0.08)
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.embed.weight.data.uniform_(-0.08, 0.08)
        self.input_proj.weight.data.uniform_(-0.08, 0.08)
        self.input_proj.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.08, 0.08)
        self.linear.bias.data.fill_(0)

    def forward(self, video_features, flengths, captions, lengths):
        """Decode image feature vectors and generates captions."""
        """
        :param video_features:
            video encoder output hidden states of size batch_size x max_enc_steps x hidden_dim
        :param flengths:
            video frames length of size batch_size
        :param captions:
            input target captions of size batch_size x max_dec_steps
        :param lengths:
            input captions lengths of size batch_size

        """
        # print features.size(), captions.size(), self.embed_size
        # print 'Input features, captions, lengths', features.size(), captions.size(), lengths, np.sum(lengths)
        # appending <start> token to the input captions
        batch_size,step_size = captions.shape
        max_enc_steps = video_features.shape[1]
        context_mask = rnn_mask(flengths,max_enc_steps)
        captions = torch.cat((Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda(),captions), 1)
        embeddings = self.embed(captions)
        hidden_output = Variable(torch.FloatTensor(batch_size,self.hidden_size).zero_()).cuda()
        state = None
        outputs = []
        for i in range(step_size):
            c_t, alpha = self.atten(hidden_output, video_features, context_mask)
            inp = torch.cat((embeddings[:,i,:], c_t), 1).unsqueeze(1)
            #inp = self.input_proj(inp)
            hidden_output,state = self.lstm(inp,state)
            hidden_output = hidden_output.squeeze(1)
            outputs.append(hidden_output)

        outputs = torch.transpose(torch.stack(outputs), 0, 1) # converting from step_size x batch_size x hidden_size to batch_size x step_size x hidden_size
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        outputs = self.linear(outputs)

        return outputs


    def sample(self, video_features, flengths, max_len=30, state=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        state = None
        batch_size, _, _ = video_features.shape
        max_enc_steps = video_features.shape[1]
        context_mask = rnn_mask(flengths, max_enc_steps)
        hidden_output = Variable(torch.FloatTensor(batch_size,self.hidden_size).zero_()).cuda()
        inputs = self.embed(Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda()).squeeze(1)
        for i in range(max_len + 1):  # maximum sampling length
            c_t, alpha = self.atten(hidden_output, video_features, context_mask)
            inp = torch.cat((inputs, c_t), 1).unsqueeze(1)
            #inp = self.input_proj(inp)
            hidden_output,state = self.lstm(inp,state)
            hidden_output = hidden_output.squeeze(1)
            output = self.linear(hidden_output)  # (batch_size, vocab_size)
            predicted = output.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()

    def beam_search(self, video_features, flengths, max_len=20, beam_size=5):
        """ Beam search Implementation during Inference"""
        prev_state = None
        outputs = []
        batch_size, max_enc_steps, _ = video_features.shape
        context_mask = rnn_mask(flengths, max_enc_steps)
        hidden_output = Variable(torch.FloatTensor(batch_size,self.hidden_size).zero_()).cuda()
        inputs = self.embed(Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda()).squeeze(1)
        # handle the zero step case seperately
        c_t, alpha = self.atten(hidden_output, video_features, context_mask)
        inp = torch.cat((inputs,c_t),1).unsqueeze(1)
        next_hidden, next_state = self.lstm(inp, prev_state)
        next_hidden = next_hidden.squeeze(1)
        output = self.linear(next_hidden)
        output = F.softmax(output,1)
        next_probs, next_words = torch.topk(output,beam_size)
        prev_words = torch.t(next_words)
        prev_state = []
        prev_hidden = []
        #print next_state

        for i in range(beam_size):
            prev_state.append(next_state)
            prev_hidden.append(next_hidden)
        #print prev_state
        all_probs = next_probs.cpu().data.numpy()

        generated_sequence = np.zeros((batch_size,beam_size,max_len),dtype=np.int32)
        generated_sequence[:,:,0] = next_words.cpu().data.numpy()

        # variables for final results storing
        final_results = np.zeros((batch_size,beam_size,max_len), dtype=np.int32)
        final_all_probs = np.zeros((batch_size,beam_size))
        final_results_counter = np.zeros((batch_size),dtype=np.int32) # to check the overflow of beam in fina results


        for i in range(1,max_len):
            probs = []
            state = []
            hidden = []
            words = []

            for j in range(beam_size):
                inputs = self.embed(prev_words[j])
                #print inputs
                c_t, alpha = self.atten(prev_hidden[j], video_features, context_mask)
                inp = torch.cat((inputs,c_t),1).unsqueeze(1)
                next_hidden, next_state = self.lstm(inp, prev_state[j])
                next_hidden = next_hidden.squeeze(1)
                output = self.linear(next_hidden)
                output = F.softmax(output,1)
                next_probs, next_words = torch.topk(output, beam_size)
                probs.append(next_probs)
                words.append(next_words)
                state.append(next_state)
                hidden.append(next_hidden)


            probs = np.transpose(np.array(torch.stack(probs).cpu().data.numpy()),(1,0,2))
            #state = np.transpose(np.array(state.cpu().data.numpy()),(1,0,2))
            hidden = np.transpose(np.array(torch.stack(hidden).cpu().data.numpy()),(1,0,2))
            words = np.transpose(np.array(torch.stack(words).cpu().data.numpy()),(1,0,2))
            state = [torch.cat(s,0) for s in state]
            state = torch.stack(state)
            #print state

            prev_state = []
            prev_words = []
            prev_hidden = []
            
            for k in range(batch_size):
                probs[k] = np.transpose(np.transpose(probs[k])*all_probs[k]) # multiply each beam words with each beam probs so far
                top_indices = top_n_indexes(probs[k],beam_size)
                beam_idx,top_choice_idx = zip(*top_indices)
                all_probs[k] = (probs[k])[beam_idx,top_choice_idx]
                prev_state.append([state[idx,:,k,:] for idx in beam_idx])
                prev_hidden.append([hidden[k,idx,:] for idx in beam_idx])
                prev_words.append([words[k,idx,idy] for idx,idy in top_indices])
                generated_sequence[k] = generated_sequence[k,beam_idx,:]
                generated_sequence[k,:,i] = [words[k,idx,idy] for idx,idy in top_indices]



                # code to extract complete summaries ending with [EOS] or [STOP] or [END]

                for beam_idx in range(beam_size):
                    if generated_sequence[k,beam_idx,i] == 2 and final_results_counter[k]<beam_size: # [EOS] or [STOP] or [END] word / check overflow
                        # print generated_sequence[k,beam_idx]
                        final_results[k,final_results_counter[k],:] = generated_sequence[k,beam_idx,:]
                        final_all_probs[k,final_results_counter[k]] = all_probs[k,beam_idx]
                        final_results_counter[k] += 1 
                        all_probs[k,beam_idx] = 0.0 # supress this sentence to flow further through the beam


            if np.sum(final_results_counter) == batch_size*beam_size: # when suffiecient hypothsis are obtained i.e. beam size hypotheis, break the process
                # print "Encounter a case"
                break

            # transpose batch to usual
            #print prev_state
            prev_state = [torch.stack(s,0) for s in prev_state]
            prev_state = torch.stack(prev_state,0)
            prev_state = torch.transpose(prev_state,0,1)
            tmp_state = torch.transpose(prev_state,1,2)
            prev_state = []
            for k in range(beam_size):
                prev_state.append(tuple((tmp_state[k,0,:,:].unsqueeze(0).contiguous(),tmp_state[k,1,:,:].unsqueeze(0).contiguous())))

            #print prev_state
            prev_words = np.transpose(np.array(prev_words),(1,0)) # set order [beam_size, batch_size]
            prev_words = Variable(torch.LongTensor(prev_words)).cuda()
            prev_hidden = np.transpose(np.array(prev_hidden),(1,0,2))
            prev_hidden = Variable(torch.FloatTensor(prev_hidden)).cuda()
            #print prev_hidden[0]
            #print prev_state[0]
            #print generated_sequence
            


        sampled_ids = []
        for k in range(batch_size):
            avg_log_probs = []
            for j in range(beam_size):
                try:
                    num_tokens = final_results[k,j,:].tolist().index(2)+1 #find the stop word and get the lenth of the sequence based on that
                except:
                    num_tokens = 1 # this case is when the number of hypotheis are not equal to beam size, i.e., durining the process sufficinet hypotheisis are not obtained
                probs = np.log(final_all_probs[k][j])/num_tokens

                avg_log_probs.append(probs)
            avg_log_probs = np.array(avg_log_probs)
            sort_order = np.argsort(avg_log_probs)
            sort_order[:] = sort_order[::-1]
            sort_generated_sequence  = final_results[k,sort_order,:]
            sampled_ids.append(sort_generated_sequence[0])
            #print sort_generated_sequence


        return np.asarray(sampled_ids)


    def rl_sample(self, video_features, flengths, max_len=20, sampling='multinomial'):
        sampled_ids = []
        state = None
        outputs = []
        batch_size, max_enc_steps, _ = video_features.shape
        context_mask = rnn_mask(flengths,max_enc_steps)
        hidden_output = Variable(torch.FloatTensor(batch_size,self.hidden_size).zero_()).cuda()
        inputs = self.embed(Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda()).squeeze(1)
        for i in range(max_len):  # maximum sampling length
            c_t, alpha = self.atten(hidden_output, video_features, context_mask)
            inp = torch.cat((inputs, c_t), 1).unsqueeze(1)
            # inp = self.input_proj(inp)
            hidden_output,state = self.lstm(inp,state)
            hidden_output = hidden_output.squeeze(1)
            output = self.linear(hidden_output)  # (batch_size, vocab_size)
            outputs.append(output)
            prob = F.softmax(output, 1)
            if sampling == 'multinomial':
                predicted = torch.multinomial(prob, 1)
                predicted = predicted.squeeze(1)
            elif sampling == 'argmax':
                predicted = prob.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        outputs = torch.transpose(torch.stack(outputs), 0, 1) 
        return sampled_ids.squeeze(), outputs



class Attention(nn.Module):
    def __init__(self, args, enc_dim, dec_dim, attn_dim=None):
        super(Attention, self).__init__()
        
        self.args = args
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim
        if self.args.birnn:
            self.birnn = 2
        else:
            self.birnn = 1

        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=True)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.attn_linear = nn.Linear(self.attn_dim, 1, bias=False)
        self.init_weights()
 

    def init_weights(self):
        self.encoder_in.weight.data.uniform_(-0.08, 0.08)
        self.encoder_in.bias.data.fill_(0)
        self.decoder_in.weight.data.uniform_(-0.08, 0.08)
        self.attn_linear.weight.data.uniform_(-0.08, 0.08)


    def forward(self, dec_state, enc_states, mask, dag=None):
        """
        :param dec_state: 
            decoder hidden state of size batch_size x dec_dim
        :param enc_states:
            all encoder hidden states of size batch_size x max_enc_steps x enc_dim
        :param flengths:
            encoder video frame lengths of size batch_size
        """
        dec_contrib = self.decoder_in(dec_state)
        batch_size, max_enc_steps, _  = enc_states.size()
        enc_contrib = self.encoder_in(enc_states.contiguous().view(-1, self.enc_dim)).contiguous().view(batch_size, max_enc_steps, self.attn_dim)
        pre_attn = F.tanh(enc_contrib + dec_contrib.unsqueeze(1).expand_as(enc_contrib))
       
        
        energy = self.attn_linear(pre_attn.view(-1, self.attn_dim)).view(batch_size, max_enc_steps)
        alpha = F.softmax(energy, 1)
        # mask alpha and renormalize it
        alpha = alpha* mask
        alpha = torch.div(alpha, alpha.sum(1).unsqueeze(1).expand_as(alpha))

        context_vector = torch.bmm(alpha.unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)

        return context_vector, alpha






