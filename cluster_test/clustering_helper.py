class ClusteringAnalyzer(object):
    """
    聚类结果分析器
    """
    def __init__(self, clustering_result, n_classes):
        """
        :param clustering_result: 保存聚类结果,格式为tuple(api id, 原cate id, 簇id)
        :param n_classes: 共有多少个类
        """
        self._clustering_result = clustering_result       # 保存聚类结果,格式为tuple(api id, 原cate id, 簇id)
        self.pred_counter = [0 for _ in range(n_classes)]  # 统计各个类共有多少个api
        self._num_class = n_classes
        self._result_label = [[] for _ in range(n_classes)]
        for item in self._clustering_result:        # 按照原cate id分类统计聚类结果, 只保留api id
            self._result_label[item[1]].append(item[0])
        self._result_pred = [[] for _ in range(n_classes)]
        for item in self._clustering_result:        # 按照聚类结果的簇id来分类统计聚类结果, 只保留api id
            self._result_pred[item[2]].append(item[0])
        # print(self.result_pred)
        # print(self.result_label)
        self._intersection = [[0 for _ in range(self._num_class)] for _ in range(self._num_class)]

    def analysis(self):
        print(self._clustering_result)
        for item in self._clustering_result:
            self.pred_counter[item[2]] += 1
        print(self.pred_counter)
        for i in range(self._num_class):
            for j in range(self._num_class):
                intersection_i_j = 0
                for api_id in self._result_label[i]:
                    if api_id in self._result_pred[j]:
                        intersection_i_j += 1
                self._intersection[i][j] = intersection_i_j
        print("purity:", self._compute_purity())
        print("recall:", self._compute_recall())

    def _compute_purity(self):
        """
        compute 'purity' for this clustering result
        'i' mean the category id from original label
        'j' mean the category id from the clustering experimental
        """
        purity = 0.0
        for j in range(self._num_class):
            num_intersection = 0
            for i in range(self._num_class):
                num_intersection = max(num_intersection, self._intersection[i][j])
            purity += num_intersection / len(self._clustering_result)
        return purity

    def _compute_recall(self):
        """
        compute 'recall' for this clustering result
        'i' mean the category id from original label
        'j' mean the category id from the clustering experimental
        """
        recall = 0.0
        for i in range(self._num_class):
            num_intersection = 0
            for j in range(self._num_class):
                num_intersection = max(num_intersection, self._intersection[i][j])
            recall += num_intersection / len(self._clustering_result)
        return recall
