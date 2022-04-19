import argparse
from train import Train
from test import Test


parser = argparse.ArgumentParser()
#parser.add_argument('-gpu', required=True, default="1", help="# of gpu")
#parser.add_argument('-option', required=True, default='BASE', help="BASE / LR / CT ")
#parser.add_argument('-task', required=True, default='TRAIN', help="TRAIN / TEST ")
parser.add_argument('-gpu', default="3", help="# of gpu")
parser.add_argument('-option', default='BASE', help="BASE / LR / CT ")
parser.add_argument('-task',  default='TRAIN', help="TRAIN / TEST ")

#parser.add_argument('-data_task', default="CF", help="task of dataset[MT / CF]" )
parser.add_argument('-data_task', default="MT", help="task of dataset[MT / CF]" )

#parser.add_argument('-data_pkl', default="data_imdb_256_CF.pickle", help="file name of preprocessed data(pickle file)")
parser.add_argument('-data_pkl', default="data_wmt16_512.pickle", help="file name of preprocessed data(pickle file)")
#parser.add_argument('-model_save', default="classification_16.pt", help="name of saved(updated) model")
parser.add_argument('-model_save', default="test_model.pt", help="name of saved(updated) model")
#parser.add_argument('-pred_save', default="LR1e-3_16_E.txt", help="name of file containing prediction result")
parser.add_argument('-pred_save', default="test_save.txt", help="name of file containing prediction result")

parser.add_argument('-batch_size', default=16, type=int, help="batch size")
parser.add_argument('-n_epoch', default=30, type=int, help="# of epoch")
parser.add_argument('-learning_rate', default=1e-4, type=float, help="learning rate") #1e-5 for test
parser.add_argument('-num_warmup', default=4000, type=int, help="# of warmup (about learning rate)") # 2000으로 test (transformer_lr1e-3_64.pt)

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
data_task = opt.data_task
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
if task == "TRAIN":
    Train(gpu, option, batch_size, n_epoch, data_pkl, model_save, learning_rate, num_warmup, hidden_dim, n_layer, n_head, ff_dim, dropout, data_task)
if task == "TEST":
    Test(gpu, option, batch_size, data_pkl, model_save, pred_save, hidden_dim, n_layer, n_head, ff_dim, dropout, data_task)