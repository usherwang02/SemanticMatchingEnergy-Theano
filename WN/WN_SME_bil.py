#! /usr/bin/python

from WN_exp import *
from WN_evaluation import *
launch(op='SME_lin', simfn='Dot', ndim=100, nhid=50, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=1000, test_all=100,  Nent=1640,
        Nsyn=1635, Nrel=5,loademb=False,loadmodel=False,
    savepath='/Users/a/Downloads/SME-master2/data/increment/test/original/SME-lin/', datapath='/Users/a/Downloads/SME-master2/data/increment/test/original/')
# Training takes 4 hours on GTX675M and an intel core i7 processor

# print "\n##### EVALUATION #####\n"
#
# ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='/Users/a/Downloads/SME-master2/data/WN_SME_bil/best_valid_model.pkl')
# RankingEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='/Users/a/Downloads/SME-master2/data/WN_SME_bil/best_valid_model.pkl')
