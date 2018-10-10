#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cPickle as pickle
f = open('/Users/a/WN_SME_lin/best_valid_model.pkl')
infos = pickle.load(f)
f.close()
for info in infos:
    text.append(info)
with open('/Users/a/Downloads/SME-master2/WN/best_valid_model.txt', 'w') as fo1:
    fo1.writelines(text)
#print info   #show file
