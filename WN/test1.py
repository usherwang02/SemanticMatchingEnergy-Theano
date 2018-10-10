#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model import *
import json
import MySQLdb
import pickle
db = MySQLdb.connect("localhost", "root", "usher0202", "Service_Recommendation", charset='utf8')
# 使用cursor()方法获取操作游标
cursor = db.cursor()
def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs
triple_lack_left = []
triple_lack_right = []
f = open('/Users/a/Downloads/RANKFILE/train.txt', 'r')
dat = f.readlines()
f.close()
for i in dat:
    lhs, rel, rhs = parseline(i[:-1])
    triple_lack_left.append(rel[0]+rhs[0])
    triple_lack_right.append(rel[0]+lhs[0])
f_decre = open('/Users/a/Downloads/RANKFILE/SME_bil/top10-left.json', 'r')
rank_left = json.load(f_decre)
f_decre.close()
f_decre = open('/Users/a/Downloads/RANKFILE/SME_bil/top10-right.json', 'r')
rank_right = json.load(f_decre)
f_decre.close()
# f = open('/Users/a/Downloads/embedKB-master/data/NeuralTensorNetwork/entity2id.json', 'r')
# entity2id = json.load(f)
# entity2id_values = list(entity2id.values())
# entity2id_keys = list(entity2id.keys())
# id2entity = {}
# for i in entity2id_values:
#     id2entity[i] = entity2id_keys[i]
j=0
f = open('/Users/a/Downloads/RANKFILE/SME_lin/WN_idx2synset.pkl', 'r')
id2entity = pickle.load(f)
# print len(triple_lack_left)
for e in rank_left:
    entity_list = ""
    # print ("##########")
    for entity in e:
        # print entity
        entity_list += id2entity[entity].encode("utf-8")
        entity_list += ','
    entity_list = entity_list.rstrip(',')
    sql = "INSERT INTO SME_bil_left(triple, entity_rank) VALUES ('%s','%s')" % (triple_lack_left[j], entity_list)
    # print type(triple_lack_left[j])
    # print entity_list
    j += 1
    try:
    # 执行sql语句
        cursor.execute(sql)
    # 提交到数据库执行
        db.commit()
    except:
    # Rollback in case there is any error
        print "wrong:" + sql
        db.rollback()
j=0
for e in rank_right:
    entity_list = ""
    # print ("##########")
    for entity in e:
        entity_list += id2entity[entity].encode("utf-8")
        entity_list += ','
    entity_list = entity_list.rstrip(',')
    sql = "INSERT INTO SME_bil_right(triple, entity_rank) VALUES ('%s','%s')" % (triple_lack_right[j], entity_list)
    # print type(triple_lack_left[j])
    # print entity_list
    j += 1
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # Rollback in case there is any error
        print "wrong:" + sql
        db.rollback()

# try:
#     sql = "SELECT * FROM NTN_left"
#     cursor.execute(sql)
#     results = cursor.fetchall()
#     for row in results:
#         print row[0]
# except:
#     print "Error: unable to fecth data"

