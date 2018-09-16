# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm  #matplotlib中文显示

def load_data():
    filename = "evaluation_results.txt"
    r1 = []
    r2 = []
    r3 = []
    with open(filename,mode='r', encoding="utf-8") as f:
        i = 1
        for line in f.readlines():
            line_div = line.strip()
            if i <= 2:
                r1.append(line_div)
                i += 1
            elif i <= 4:
                r2.append(line_div)
                i += 1
            elif i <= 6:
                r3.append(line_div)
                i += 1
    return r1,r2,r3

if __name__ == '__main__':
    result = load_data()
    #zhfont1 = fm.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')

    plt.xlabel("评估类型")#fontproperties=zhfont1
    plt.ylabel("评估数值")
    plt.title("聚类效果评估")

    width = 3
    kind = ['Purity','Recall']
    x1 = [10, 20]
    x2 = [10 + width, 20 + width]
    x3 = [10 + width + width,20 + width + width]

    plt.bar(x1, result[0], facecolor='red', width=3, label='TF-IDF')
    plt.bar(x2, result[1], facecolor='green', width=3, label='LDA',tick_label = kind)
    plt.bar(x3, result[2], facecolor='yellow', width=3, label='LDA+KMeans')


    plt.legend()

    plt.show()









