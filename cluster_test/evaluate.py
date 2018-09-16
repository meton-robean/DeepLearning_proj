import collections
import math


def purity(result, label):
    # 纯度

    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:  # 求交集
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)

    return sum(t) / total_num


def NMI(result, label):
    # 标准化互信息

    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)

    # 计算互信息量
    MI = 0
    eps = 1.4e-45  # 取一个很小的值来避免log 0

    for k in cluster_counter:
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:
                    count += 1
            p_k = 1.0 * cluster_counter[k] / total_num
            p_j = 1.0 * original_counter[j] / total_num
            p_kj = 1.0 * count / total_num
            MI += p_kj * math.log(p_kj / (p_k * p_j) + eps, 2)

    # 标准化互信息量
    H_k = 0
    for k in cluster_counter:
        H_k -= (1.0 * cluster_counter[k] / total_num) * math.log(1.0 * cluster_counter[k] / total_num + eps, 2)
    H_j = 0
    for j in original_counter:
        H_j -= (1.0 * original_counter[j] / total_num) * math.log(1.0 * original_counter[j] / total_num + eps, 2)

    NMI_value = 2.0 * MI / (H_k + H_j)
    return NMI_value


def contingency_table(result, label):
    total_num = len(label)

    TP = TN = FP = FN = 0
    for i in range(total_num):
        for j in range(i + 1, total_num):
            if label[i] == label[j] and result[i] == result[j]:
                TP += 1
            elif label[i] != label[j] and result[i] != result[j]:
                TN += 1
            elif label[i] != label[j] and result[i] == result[j]:
                FP += 1
            elif label[i] == label[j] and result[i] != result[j]:
                FN += 1
    return (TP, TN, FP, FN)


def rand_index(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * (TP + TN) / (TP + FP + FN + TN)


def precision(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * TP / (TP + FP)


def recall(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0 * TP / (TP + FN)


def F_measure(result, label, beta=1):
    prec = precision(result, label)
    r = recall(result, label)
    return (beta * beta + 1) * prec * r / (beta * beta * prec + r)