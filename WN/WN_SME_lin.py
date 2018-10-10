#! /usr/bin/python

from WN_exp import *
from WN_evaluation import *

# launch(op='SME_lin', simfn='Dot', ndim=100, nhid=50, marge=1., lremb=0.01,
#     lrparam=1.0, nbatches=100, totepochs=50, test_all=50,
#     savepath='/Users/a/Downloads/SME-master2/data/increment/1w/changed/', datapath='/Users/a/Downloads/SME-master2/data/increment/1w/changed/', dataset='WN', Nent=23548,
#         Nsyn=23543, Nrel=5,
#        loadmodel='/Users/a/Downloads/SME-master2/data/increment/1w/original/best_valid_model.pkl',
#        # loadmodel=False,
#            loademb=False,
#        incre='/Users/a/Downloads/SME-master2/data/increment/1w/changed/',
#        # decre='/Users/a/Downloads/SME-master2/data/decrement/1w/test/',
#        increent=0, postion='lhs', neval=50, seed=123)
launch(op='SME_lin', simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.001,
               lrparam=1.0, nbatches=100, totepochs=50, test_all=50,
               savepath='/Users/a/Downloads/SME-master2/data/increment/test/changed/',
               datapath='/Users/a/Downloads/SME-master2/data/increment/test/changed/', dataset='WN', Nent=1643,
               Nsyn=1638, Nrel=5,
               # loadmodel='/Users/a/Downloads/SME-master2/data/increment/test/changed/best_valid_model.pkl',
               loadmodel=False,
               loademb=False,
               # incre=state.incre,
               # decre='/Users/a/Downloads/SME-master2/data/decrement/1w/changed/',
               increent=0, postion='', neval=50, seed=123)
# Training takes 4 hours on GTX675M and an intel core i7 processor

print ("\n##### EVALUATION #####\n")
#
# ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='/Users/a/Downloads/SME-master2/data/WN_SME_lin/best_valid_model.pkl')
RankingEval(datapath='/Users/a/Downloads/SME-master2/data/increment/test/changed/', loadmodel='/Users/a/Downloads/SME-master2/data/increment/test/changed/best_valid_model.pkl', Nsyn=1643)
# define the dimensions for the embeddings in the model

