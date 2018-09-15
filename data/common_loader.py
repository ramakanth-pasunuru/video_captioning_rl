from __future__ import print_function
import json
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
np.random.seed(111)

#sys.path.append('/playpen1/home/ram/video_caption_eval')
#from automatic_evaluation import evaluate


PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[END]' # This has a vocab id, which is used at the end of untruncated target sequences






class Vocab(object):
    def __init__(self,vocab_file,max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. 
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.
                
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f.read().splitlines():
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self,text,max_sentence_length, add_start_token=False):
        text = text + ' ' + STOP_DECODING
        if add_start_token:
            text = START_DECODING + ' ' + text
        tokens = []
        seq_length = 0
        for word in text.split()[:max_sentence_length]: # also need one more for [END] token
            tokens.append(self.word2id(word))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return np.asarray(tokens),seq_length


class Batch(object):

    def __init__(self):
        self._dict = {}


    def put(self,key,value):
        if self._dict.get(key) is None:
            self._dict[key] = value
        else:
            raise Exception("key:{} already exits".format(key))

    def get(self,key):
       if self._dict.get(key) is not None:
           return self._dict[key]
       else:
           raise Exception("key:{} already exits".format(key))



class MSRVTTBatcher(object):

    def __init__(self,hps,mode,vocab):
        
        self._vid_feature_path = hps.vid_feature_path
        self._captions_path = hps.captions_path
        self._max_enc_steps = hps.encoder_rnn_max_length
        self._max_dec_steps = hps.decoder_rnn_max_length
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self._vid_dim = hps.vid_dim
        self.data,self.data_dict = self._process_data()
        self.num_steps = int(len(self.data)/self._batch_size) + 1

    def _process_data(self):
        """this module extracts data from videos and caption files and creates batches"""
        # load json data which contains all the information
        data = []
        data_dict = {}
        filename ='sents_'+self._mode+'.txt'
        with open(os.path.join(self._captions_path,filename),'r') as f:
            for line in f.read().splitlines():
                line = line.split('\t')
                vid_id = line[0]
                caption = line[1]
                data.append((vid_id,caption))
                if data_dict.get(vid_id) is None:
                    data_dict[vid_id] = [caption]
                else:
                    data_dict[vid_id].append(caption)
      
        if self._mode == 'train':
            np.random.shuffle(data)
        else:
            data,_ = zip(*data) # consider only video ids for evaluation
            data = sorted(set(data),key=data.index)

        return data,data_dict

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch, original_caption):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices] 
    
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)
        
        for i in range(0,len(self.data),self._batch_size):
            start = i
            if i+self._batch_size > len(self.data): # handling leftovers
                end = len(self.data)
                current_batch_size = end-start
            else:
                end = i+self._batch_size
                current_batch_size = self._batch_size
            if self._mode == 'train':
                video_id,original_caption = zip(*self.data[start:end])
            else:
                video_id = self.data[start:end]

            video_features = [np.load(os.path.join(self._vid_feature_path,key+'.mp4.npy')) for key in video_id]
            
            if self._mode == 'train':
                caption_batch = []
                caption_length = []
                for cap in original_caption:
                    cap_id,cap_length = self.vocab.texttoidx(cap,self._max_dec_steps)
                    caption_batch.append(cap_id)
                    caption_length.append(cap_length)

            original_caption_dict = {}
            for vid in video_id:
                original_caption_dict[vid] = self.data_dict[vid]
                
            # transform/clip frames
            video_batch = np.zeros((current_batch_size,self._max_enc_steps,self._vid_dim))
            video_length = []
            for idx,feat in enumerate(video_features):
                if len(feat)>self._max_enc_steps:
                    video_batch[idx][:] = feat[:self._max_enc_steps]
                    video_length.append(self._max_enc_steps)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            if self._mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length, original_caption = self.sort_based_on_caption_lengths(
                                                                                                            np.asarray(video_batch), np.asarray(video_length), 
                                                                                                            np.asarray(video_id), np.asarray(caption_batch), 
                                                                                                            np.asarray(caption_length), np.asarray(original_caption)) 

            else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)


            batch = Batch()
            if self._mode == 'train':
                batch.put('original_caption',original_caption)
                batch.put('caption_batch',torch.LongTensor(caption_batch))
                batch.put('caption_len_batch',caption_length)
            batch.put('original_caption_dict',original_caption_dict)
            batch.put('video_batch',torch.FloatTensor(video_batch))
            batch.put('video_len_batch',video_length)
            batch.put('video_id',video_id)
            yield batch




class SNLIBatcher(object):

    def __init__(self,max_steps,vocab):
        self._max_steps = max_steps
        self.vocab = vocab


    def process_external_data(self, prem, hypo):

        original_premise = prem
        original_hypothesis = hypo


        premise_batch = []
        premise_length = []
        hypothesis_batch = []
        hypothesis_length = []
        
        for prem, hypo in zip(original_premise, original_hypothesis):

            prem_id, prem_length = self.vocab.texttoidx(prem, self._max_steps, add_start_token=True)
            hypo_id, hypo_length = self.vocab.texttoidx(hypo, self._max_steps, add_start_token=True)
            premise_batch.append(prem_id)
            premise_length.append(prem_length)
            hypothesis_batch.append(hypo_id)
            hypothesis_length.append(hypo_length)



        batch = Batch()
        batch.put('original_premise', original_premise)
        batch.put('original_hypothesis', original_hypothesis)
        batch.put('premise_batch', torch.LongTensor(np.asarray(premise_batch)))
        batch.put('premise_length', np.asarray(premise_length))
        batch.put('hypothesis_batch', torch.LongTensor(np.asarray(hypothesis_batch)))
        batch.put('hypothesis_length', np.asarray(hypothesis_length))

        return batch

