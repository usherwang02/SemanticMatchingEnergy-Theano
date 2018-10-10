#! /usr/bin/python

from WN_exp import *
from WN_evaluation import *

print "\n----- SE -----\n"

launch(op='SE', simfn='L1', ndim=2, nhid=3, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=2, test_all=1, savepath='test',
    neval=10, datapath='/Users/a/Downloads/SME-master2/data/')

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl')
RankingEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl',
        neval=50)

print "\n----- Unstructured -----\n"

launch(op='Unstructured', simfn='Dot', ndim=2, nhid=3, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=2, test_all=1, savepath='test',
    neval=10, datapath='/Users/a/Downloads/SME-master2/data/')

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl')
RankingEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl',
        neval=50)

print "\n----- SME_lin -----\n"

launch(op='SME_lin', simfn='Dot', ndim=2, nhid=3, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=2, test_all=1, savepath='test',
    neval=10, datapath='/Users/a/Downloads/SME-master2/data/')

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl')
RankingEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl',
        neval=50)

print "\n----- SME_bil -----\n"

launch(op='SME_bil', simfn='Dot', ndim=2, nhid=3, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=2, test_all=1, savepath='test',
    neval=10, datapath='/Users/a/Downloads/SME-master2/data/')

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl')
RankingEval(datapath='/Users/a/Downloads/SME-master2/data/', loadmodel='test/best_valid_model.pkl',
        neval=50)
