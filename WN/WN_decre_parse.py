import os
import pickle

import numpy as np
import scipy.sparse as sp
from model import *
import numpy
def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)
def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

datapath = '/Users/a/Downloads/SME-master2/data/decrement/1w/original/'
savepath = '/Users/a/Downloads/SME-master2/data/decrement/1w/changed/'
position = 'lhs'
fo = open(datapath + 'WN_synset2idx.pkl', 'r')
synset2idx = pickle.load(fo)
fo.close()
# print synset2idx
f2 = open(savepath + 'decre-entity.txt', 'r')
dat = f2.readlines()
f2.close()
trainl = load_file(datapath + 'WN-train-lhs.pkl')
trainr = load_file(datapath + 'WN-train-rhs.pkl')
trainrel = load_file(datapath + 'WN-train-rel.pkl')
arrl1, arrl2 = trainl.nonzero()
arrr1, arrr2 = trainr.nonzero()
arrrel1, arrrel2 = trainrel.nonzero()
f1 = open(savepath + 'train-lhs.txt', 'a')
f2 = open(savepath + 'train-rhs.txt', 'a')
f3 = open(savepath + 'train-rel.txt', 'a')
for datatyp in ['train']:
    f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    whole_train_triple = f.readlines()
    f.close()


inpll = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inplr = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inplo = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inprl = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inprr = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inpro = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inpol = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inpor = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
inpoo = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(whole_train_triple)),
                     dtype='float32')
num_list = []
entity_list = []
for i in dat:
    decre_entity = i.rstrip('\n')
    entity_list.append(synset2idx[decre_entity])
    if synset2idx[decre_entity] in arrl1:
        tripule_lhs = arrl2[numpy.where(arrl1 == synset2idx[decre_entity])].tolist()
        for num in tripule_lhs:
            f1.write(whole_train_triple[num])
            lhs, rel, rhs = parseline(whole_train_triple[num][:-1])
            num_list.append(whole_train_triple[num])
            inpll[synset2idx[lhs[0]], num] = 1
            inplr[synset2idx[rhs[0]], num] = 1
            inplo[synset2idx[rel[0]], num] = 1
    if synset2idx[decre_entity] in arrr1:
        tripule_rhs = (arrr2[numpy.where(arrr1 == synset2idx[decre_entity])].tolist())
        for num in tripule_rhs:
            f2.write(whole_train_triple[num])
            lhs, rel, rhs = parseline(whole_train_triple[num][:-1])
            num_list.append(whole_train_triple[num])
            inprl[synset2idx[lhs[0]], num] = 1
            inprr[synset2idx[rhs[0]], num] = 1
            inpro[synset2idx[rel[0]], num] = 1
    if synset2idx[decre_entity] in arrrel1:
        tripule_rel = (arrrel2[numpy.where(arrrel1 == synset2idx[decre_entity])].tolist())
        for num in tripule_rel:
            f3.write(whole_train_triple[num])
            lhs, rel, rhs = parseline(whole_train_triple[num][:-1])
            num_list.append(whole_train_triple[num])
            inpol[synset2idx[lhs[0]], num] = 1
            inpor[synset2idx[rhs[0]], num] = 1
            inpoo[synset2idx[rel[0]], num] = 1
f1.close()
f2.close()
f3.close()

f_new = open(savepath + 'new_wordnet-mlj12-%s.txt' % datatyp,"w")
for line in whole_train_triple:
    if line in num_list:
        continue
    f_new.write(line)
f_new.close()


f = open(savepath + 'decre-lhs-lhs.pkl', 'w')
g = open(savepath + 'decre-lhs-rhs.pkl', 'w')
h = open(savepath + 'decre-lhs-rel.pkl', 'w')
i = open(savepath + 'decre-rhs-lhs.pkl', 'w')
j = open(savepath + 'decre-rhs-rhs.pkl', 'w')
k = open(savepath + 'decre-rhs-rel.pkl', 'w')
l = open(savepath + 'decre-rel-lhs.pkl', 'w')
m = open(savepath + 'decre-rel-rhs.pkl', 'w')
n = open(savepath + 'decre-rel-rel.pkl', 'w')
pickle.dump(inpll.tocsr(), f, -1)
pickle.dump(inplr.tocsr(), g, -1)
pickle.dump(inplo.tocsr(), h, -1)
pickle.dump(inprl.tocsr(), i, -1)
pickle.dump(inprr.tocsr(), j, -1)
pickle.dump(inpro.tocsr(), k, -1)
pickle.dump(inpol.tocsr(), l, -1)
pickle.dump(inpor.tocsr(), m, -1)
pickle.dump(inpoo.tocsr(), n, -1)
f.close()
g.close()
h.close()
i.close()
j.close()
k.close()
l.close()
m.close()
n.close()



#################################################
### Creation of the synset/indices dictionnaries

np.random.seed(753)

synlist = []
rellist = []

for datatyp in ['train', 'valid', 'test']:
    if datatyp == 'train':
        f = open(savepath + 'new_wordnet-mlj12-%s.txt' % datatyp, 'r')
    else:
        f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        if i != '':
            lhs, rel, rhs = parseline(i[:-1])
            synlist += [lhs[0], rhs[0]]
            rellist += [rel[0]]

synset = np.sort(list(set(synlist)))
relset = np.sort(list(set(rellist)))

new_synset2idx = {}
new_idx2synset = {}

idx = 0
for i in synset:
    new_synset2idx[i] = idx
    new_idx2synset[idx] = i
    idx += 1
nbsyn = idx
print ("Number of synsets in the dictionary: ", nbsyn)
# add relations at the end of the dictionary
for i in relset:
    new_synset2idx[i] = idx
    new_idx2synset[idx] = i
    idx += 1
nbrel = idx - nbsyn
print ("Number of relations in the dictionary: ", nbrel)

f = open(savepath + 'WN_synset2idx.pkl', 'w')
g = open(savepath + 'WN_idx2synset.pkl', 'w')
pickle.dump(new_synset2idx, f, -1)
pickle.dump(new_idx2synset, g, -1)
f.close()
g.close()
f = open(savepath + 'decre_entity_list.pkl', 'w')
pickle.dump(num_list, f, -1)
f.close()
for i in num_list:
    i = i.rstrip('\n')
    lhs, rel, rhs = parseline(i)
    for j in [lhs[0], rel[0], rhs[0]]:
        if not new_synset2idx.has_key(j):
            entity_list.append(synset2idx[j])
entity_list = list(set(entity_list))
f = open(savepath + 'decre_num_list.pkl', 'w')
pickle.dump(entity_list, f, -1)
f.close()

fo = open(savepath + 'WN_synset2idx.pkl', 'r')
synset2idx = pickle.load(fo)
fo.close()
for datatyp in ['train', 'valid', 'test']:
    if datatyp == 'train':
        f = open(savepath + 'new_wordnet-mlj12-%s.txt' % datatyp, 'r')
    else:
        f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpr = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpo = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        inpl[synset2idx[lhs[0]], ct] = 1
        inpr[synset2idx[rhs[0]], ct] = 1
        inpo[synset2idx[rel[0]], ct] = 1
        ct += 1
    f = open(savepath + 'WN-%s-lhs.pkl' % datatyp, 'w')
    g = open(savepath + 'WN-%s-rhs.pkl' % datatyp, 'w')
    h = open(savepath + 'WN-%s-rel.pkl' % datatyp, 'w')
    pickle.dump(inpl.tocsr(), f, -1)
    pickle.dump(inpr.tocsr(), g, -1)
    pickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

