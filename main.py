import argparse
from train import Train
from test import Test


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', required=True, help="# of gpu")
parser.add_argument('-option', required=True, help="Transformer / linformer / mixture model ")
parser.add_argument('-task', required=True, help="train / test ")

parser.add_argument('-data_pkl', default="data.pickle", help="file name of preprocessed data(pickle file)")
parser.add_argument('-model_save', default="transformer_e2g.pt", help="name of saved(updated) model")
parser.add_argument('-pred_save', default="pred.txt", help="name of file containing prediction result")

parser.add_argument('-batch_size', default=64, type=int, help="batch size")
parser.add_argument('-n_epoch', default=50, type=int, help="# of epoch")
parser.add_argument('-learning_rate', default=1e-4, type=int, help="learning rate")
parser.add_argument('-num_warmup', default=8000, type=int, help="# of warmup (about learning rate)")

parser.add_argument('-hidden_dim', type=int, default= 512, help="hidden dimension(=d_model")
parser.add_argument('-n_layer', type=int, default=  6, help="# of encoder&decoder layer")
parser.add_argument('-n_head', type=int, default= 8, help="# of head(about multi-head attention)")
parser.add_argument('-ff_dim', type=int, default= 2048, help="dimension of feed-forward neural network")
parser.add_argument('-dropout', type=float,default= 0.1, help="ratio of dropout")

opt = parser.parse_args()

#==========================================#
gpu = opt.gpu
option = opt.option
task = opt.task
batch_size = opt.batch_size
n_epoch = opt.n_epoch
data_pkl = opt.data_pkl
model_save = opt.model_save
learning_rate = opt.learning_rate
num_warmup = opt.num_warmup
hidden_dim = opt.hidden_dim
n_layer = opt.n_layer
n_head = opt.n_head
ff_dim = opt.ff_dim
dropout = opt.dropout

pred_save = opt.pred_save
#==========================================#
if task == "train":
    Train(gpu, opt, batch_size, n_epoch, data_pkl, model_save, learning_rate, num_warmup, hidden_dim, n_layer, n_head, ff_dim, dropout)
if task == "test":
    Test(gpu, opt, data_pkl, model_save, pred_save, hidden_dim, n_layer, n_head, ff_dim, dropout)