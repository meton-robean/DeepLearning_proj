# coding = utf=8

from api import Api
from clustering_helper import ClusteringAnalyzer
import numpy as np
from nltk.stem import PorterStemmer
import time
import os
import csv
import pandas as pd
import codecs
import shutil
from sklearn import feature_extraction
import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation
import evaluate as A



description_file_name = "descriptions.txt"
category_file_name = "categories.txt"

def load_api_data_from_local_file_only_main_cate():
    '''
    将存入本地文件中的category数据和api的信息读入内存（只包含每个api的主category)
    :return:(1)元素为（category.id,category.name)类型的列表
             (2)元素为api类型的对象的列表
    '''
    cate_tuple_list = []
    api_info_list = []
    with open(description_file_name, mode='r', encoding="utf-8") as f_description_in, open(category_file_name, mode='r',
                                                                                           encoding="utf-8") as f_cate_in:
        for line in f_cate_in.readlines():
            line_div = line.strip().split(r" ")
            cate_tuple_list.append((int(line_div[0]), line_div[1]))
        num_categories = len(cate_tuple_list)  # 共有多少种category
        for line in f_description_in.readlines():
            line = line.strip()
            line_div = line.split(r"----")
            api_basic_info, categories = line_div[0], line_div[1]
            splited_infos = api_basic_info.split(r"--")
            #category_vector = [0 for i in range(num_categories)]
            #category_vector[int(categories)] = 1  # 将主category的下标单独设为一个数值
            api_info_list.append(Api(int(splited_infos[0]), splited_infos[1], splited_infos[2], categories))

    return cate_tuple_list, api_info_list

result = load_api_data_from_local_file_only_main_cate()


if __name__=='__main__':

    corpus = []  #文档语料

    for i in range(len(result[1])):
        corpus.append(result[1][i].description)

    # 构建词汇统计向量并保存，仅运行首次
    vectorizer = CountVectorizer()

    tf = vectorizer.fit_transform(corpus)
    #joblib.dump(vectorizer,'Model.pkl')

    #得到存储的vectorizer,节省预处理时间
    #vectorizer = joblib.load('Model.pkl')
    #tf = vectorizer.fit_transform(corpus)

    #LDA主题模型训练
    n_topics_ = 20
    #here n_components is set as 128 topics , then do K-mean clustering to cluster as 20 class
    lda = LatentDirichletAllocation(n_components=128,max_iter=50,learning_method='batch')
    lda.fit(tf)

    #保存LDA模型，以便后续查看
    joblib.dump(lda,'LDA.MODEL')


    doc_topic_dist = lda.transform(tf)  #doc_topic_dist is (doc_num * topics_prob_num ) matrix
    print("doc_topic_dist:\n",doc_topic_dist)


    # # 每个文本属于每一类的概率写进txt中保存下来
    # resName = "lda_results.txt"
    # ans = open(resName, 'w', encoding='utf-8')
    # for i in range(len(doc_topic_dist)):
    #     for j in range(128):
    #         ans.write(str(doc_topic_dist[i,j]) + ' ')
    #     ans.write('\n')

    # ans.close()

    #收敛效果
    ans = lda.perplexity(tf)
    print("收敛效果:\n",ans)

    # MiniBatchKmeans聚类
    from sklearn.cluster import MiniBatchKMeans
    num_cluster = 20
    clf = MiniBatchKMeans(n_clusters=num_cluster, max_iter = 1000)
    s = clf.fit(doc_topic_dist)

    #8个中心点
    print (clf.cluster_centers_)

    '''from sklearn.externals import joblib
    #仅限初次运行
    joblib.dump(clf, 'doc_cluster.pkl')
    #之后直接从保存好的模型中取出数据
    clf = joblib.load('doc_cluster.pkl')
    clusters = clf.labels_.tolist()'''

    # 每个样本所属的簇
    # label = []
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print(i, clf.labels_[i - 1])
        # label.append(clf.labels_[i-1])
        i = i + 1

    # 图形输出 降维
    print("# 图形输出 降维:")
    pca = PCA(n_components=2)
    newData = pca.fit_transform(doc_topic_dist)
    print(newData)

    i = 0
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    x10 = []
    x11 = []
    x12 = []
    x13 = []
    x14 = []
    x15 = []
    x16 = []
    x17 = []
    x18 = []
    x19 = []
    x20 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
    y8 = []
    y9 = []
    y10 = []
    y11 = []
    y12 = []
    y13 = []
    y14 = []
    y15 = []
    y16 = []
    y17 = []
    y18 = []
    y19 = []
    y20 = []

    while i < len(clf.labels_):
        if clf.labels_[i] == 0:
            x1.append(newData[i][0])
            y1.append(newData[i][1])
        elif clf.labels_[i] == 1:
            x2.append(newData[i][0])
            y2.append(newData[i][1])
        elif clf.labels_[i] == 2:
            x3.append(newData[i][0])
            y3.append(newData[i][1])
        elif clf.labels_[i] == 3:
            x4.append(newData[i][0])
            y4.append(newData[i][1])
        elif clf.labels_[i] == 4:
            x5.append(newData[i][0])
            y5.append(newData[i][1])
        elif clf.labels_[i] == 5:
            x6.append(newData[i][0])
            y6.append(newData[i][1])
        elif clf.labels_[i] == 6:
            x7.append(newData[i][0])
            y7.append(newData[i][1])
        elif clf.labels_[i] == 7:
            x8.append(newData[i][0])
            y8.append(newData[i][1])
        elif clf.labels_[i] == 8:
            x9.append(newData[i][0])
            y9.append(newData[i][1])
        elif clf.labels_[i] == 9:
            x10.append(newData[i][0])
            y10.append(newData[i][1])
        elif clf.labels_[i] == 10:
            x11.append(newData[i][0])
            y11.append(newData[i][1])
        elif clf.labels_[i] == 11:
            x12.append(newData[i][0])
            y12.append(newData[i][1])
        elif clf.labels_[i] == 12:
            x13.append(newData[i][0])
            y13.append(newData[i][1])
        elif clf.labels_[i] == 13:
            x14.append(newData[i][0])
            y14.append(newData[i][1])
        elif clf.labels_[i] == 14:
            x15.append(newData[i][0])
            y15.append(newData[i][1])
        elif clf.labels_[i] == 15:
            x16.append(newData[i][0])
            y16.append(newData[i][1])
        elif clf.labels_[i] == 16:
            x17.append(newData[i][0])
            y17.append(newData[i][1])
        elif clf.labels_[i] == 17:
            x18.append(newData[i][0])
            y18.append(newData[i][1])
        elif clf.labels_[i] == 18:
            x19.append(newData[i][0])
            y19.append(newData[i][1])
        elif clf.labels_[i] == 19:
            x20.append(newData[i][0])
            y20.append(newData[i][1])
        i = i + 1

    plt.scatter(x1, y1, c='pink')
    plt.scatter(x2, y2, c='plum')
    plt.scatter(x3, y3, c='red')
    plt.scatter(x4, y4, c='grey')
    plt.scatter(x5, y5, c='blue')
    plt.scatter(x7, y7, c='brown')
    plt.scatter(x8, y8, c='olive')
    plt.scatter(x9, y9, c='orange')
    plt.scatter(x10, y10, c='gold')
    plt.scatter(x11, y11, c='olive')
    plt.scatter(x12, y12, c='green')
    plt.scatter(x13, y13, c='tomato')
    plt.scatter(x14, y14, c='lime')
    plt.scatter(x15, y15, c='teal')
    plt.scatter(x16, y16, c='purple')
    plt.scatter(x17, y17, c='navy')
    plt.scatter(x18, y18, c='violet')
    plt.scatter(x19, y19, c='purple')
    plt.scatter(x20, y20, c='coral')

    plt.show()

    label = []
    result2 = []
    for i in range(len(result[1])):
        label.append(int(result[1][i].category))
    for j in range(len(clf.labels_)):
        result2.append(clf.labels_[j])

    purity = A.purity(result2, label)
    NMI = A.NMI(result2, label)
    TP, TN, FP, FN = A.contingency_table(result2, label)
    rand_index = A.rand_index(result2, label)
    precision = A.precision(result2, label)
    recall = A.recall(result2, label)
    F_measure = A.F_measure(result2, label)

    print("Purity:" + str(purity))
    print("Precision:" + str(precision))
    print("Recall:" + str(recall))
    print("F_measue:" + str(F_measure))


































