import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--rnn_size', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_hid', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of GCN layers')
    parser.add_argument('--rnn_type', type=str, default='gru',
                    help='rnn, gru, or lstm')
    parser.add_argument('--v_dim', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='number of layers in the RNN')
    parser.add_argument('--norm', type=str, default='weight',
                        help='number of layers in the RNN')
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=40,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=384,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='clip gradients at this value')
    parser.add_argument('--dropC', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropG', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropL', type=float, default=0.1,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropW', type=float, default=0.4,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')

    #Optimization: for the Language Model

    parser.add_argument('--optimizer', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')


    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--seed', type=int, default=777,
                    help='seed')
    parser.add_argument('--ntokens', type=int, default=777,
                    help='ntokens')

    parser.add_argument('--checkpoint_path', type=str, default='',
                    help='directory to store checkpointed models')

    parser.add_argument('--split', type=str, default='v2cp_train',
                        help='training split')
    parser.add_argument('--split_test', type=str, default='v2cp_test',
                        help='test split')

    parser.add_argument('--num_sub', type=int, default=5,
                        help='size of the proposal object set')

    parser.add_argument('--bucket', type=int, default=4,
                        help='bucket of predicted answers')

    parser.add_argument('--hint_loss_weight', type=float, default=0,
                        help='Influence strength loss weights')

    parser.add_argument('--compare_loss_weight', type=float, default=0,
                        help='self-critical loss weights')

    parser.add_argument('--reg_loss_weight', type=float, default=0.0,
                        help='regularization loss weights, set to zero in our paper ')

    parser.add_argument('--load_hint', type=float, default=0,
                        help='if load the model after using Influence strength loss')

    parser.add_argument('--use_all', type=int, default=0,
                        help='if use all QA pairs or excluding QA pairs in NUM category')

    parser.add_argument('--load_model_states', type=str, default=0,
                        help='which model to load')

    parser.add_argument('--evaluate_every', type=int, default=300,
                        help='which model to load')

    args = parser.parse_args()

    return args
