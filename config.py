import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['seq2seq'], default='seq2seq')
net_arg.add_argument('--dropout', type=float, default=0.4)
net_arg.add_argument('--weight_init', type=float, default=None)
net_arg.add_argument('--cell_type', type=str, default='lstm', choices=['lstm','gru'])
net_arg.add_argument('--birnn', type=str2bool, default=True)
net_arg.add_argument('--embed', type=int, default=512) 
net_arg.add_argument('--hid', type=int, default=1024)
net_arg.add_argument('--num_layers', type=int, default=1)
net_arg.add_argument('--vid_dim', type=int, default=4096)
net_arg.add_argument('--encoder_rnn_max_length', type=int, default=50)
net_arg.add_argument('--decoder_rnn_max_length', type=int, default=20)
net_arg.add_argument('--max_vocab_size', type=int, default=23000)
net_arg.add_argument('--max_snli_vocab_size', type=int, default=36000)
net_arg.add_argument('--beam_size', type=int, default=1)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='msrvtt')
data_arg.add_argument('--vid_feature_path', type=str, default='data/msrvtt16/msr_vtt_inceptionv4_feats_perfect')
data_arg.add_argument('--captions_path', type=str, default='data/msrvtt16/captions')
data_arg.add_argument('--vocab_file', type=str, default='data/msrvtt16/vocab')
data_arg.add_argument('--snli_vocab_file', type=str, default='data/msrvtt16/vocab_snli')

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'])
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--gamma_ml_rl', type=float, default=0.9981)
learn_arg.add_argument('--loss_function', type=str, default='xe', choices=['xe','rl', 'xe+rl'])
learn_arg.add_argument('--max_epoch', type=int, default=20)
learn_arg.add_argument('--reward_type', type=str, default='CIDEr')
learn_arg.add_argument('--grad_clip', type=float, default=10.0)
learn_arg.add_argument('--optim', type=str, default='adam')
learn_arg.add_argument('--lr', type=float, default=0.0001)
learn_arg.add_argument('--use_decay_lr', type=str2bool, default=False)
learn_arg.add_argument('--decay', type=float, default=0.96)
learn_arg.add_argument('--lambda_threshold', type=float, default=0.5)
learn_arg.add_argument('--beta_threshold', type=float, default=0.333)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_name', type=str, default='')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--load_entailment_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=1)
misc_arg.add_argument('--save_criteria', type=str, default='AVG', choices=['CIDEr', 'AVG'])
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=1111)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)


def get_args():
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed
