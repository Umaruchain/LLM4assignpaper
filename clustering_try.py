import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# 假设特征矩阵为100x200
# np.random.seed(42)  # 为了结果可重复
# X = np.random.rand(100, 200)
X = np.load('./hksts.npy')
#X = X[:35*5,:]
# 第一步：使用K-Means初始化20个簇中心
k = 35
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
centers = kmeans.cluster_centers_

# 计算每个样本到每个簇中心的距离矩阵
distance_matrix = pairwise_distances(X, centers, metric='euclidean')

# 第二步：使用PuLP求解ILP问题
# 定义问题
prob = LpProblem("Clustering_Assignment", LpMinimize)

# 定义变量
# x[i][j] = 1 如果样本i被分配到簇j，否则为0
x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(k)] for i in range(X.shape[0])]

# 目标函数：最小化总距离
prob += lpSum(distance_matrix[i][j] * x[i][j] for i in range(X.shape[0]) for j in range(k))

# 约束条件：
# 每个样本必须被分配到一个且仅一个簇
for i in range(X.shape[0]):
    prob += lpSum(x[i][j] for j in range(k)) == 1, f"One_cluster_per_sample_{i}"

# 每个簇必须恰好包含5个样本
for j in range(k):
    prob += lpSum(x[i][j] for i in range(X.shape[0])) <= 5, f"Five_samples_per_cluster_{j}_1"
    
# 每个簇必须恰好包含5个样本
for j in range(k):
    prob += lpSum(x[i][j] for i in range(X.shape[0])) >=3 , f"Three_samples_per_cluster_{j}_2"

# 求解问题
solver = PULP_CBC_CMD(msg=True)  # msg=True 显示求解过程信息
prob.solve(solver)

# 检查求解状态
from pulp import LpStatus
print(f"求解状态: {LpStatus[prob.status]}")

# 提取聚类结果
cluster_assignment = np.zeros(X.shape[0], dtype=int)
cluster_cnt = {}
for i in range(X.shape[0]):
    for j in range(k):
        if x[i][j].varValue == 1:
            cluster_assignment[i] = j
            if j not in cluster_cnt:
                cluster_cnt[j] =0 
            cluster_cnt[j]+=1
            break
        

# plt.show()
        

import pandas as pd

print(sorted(cluster_cnt.keys()))
print(len(cluster_cnt.keys()))
print(set(cluster_cnt.values()))
print(cluster_cnt)
# 原始CSV文件名
original_csv = './oral_all_demo.xlsx'

# 读取原始CSV文件
df = pd.read_excel(original_csv)

# # 检查文章数量
# if df.shape[0] < 165:
#     raise ValueError("原始CSV文件中的文章数量不足165篇。")

# 生成随机聚类结果（示例）
#num_clusters = 3
clustering_results = cluster_assignment.tolist()

# # 验证聚类结果长度
# if len(clustering_results) != 175:
#     raise ValueError("聚类结果数组的长度必须为165。")

# # 选择前165篇文章并添加聚类结果pyt
# df_subset = df.iloc[:175].copy()

df['clustering_result'] = clustering_results
# import pdb
# pdb.set_trace()

# 保存新的CSV文件
new_csv = 'oral_all_demo_with_clusterid.xlsx'
df.to_excel(new_csv, index=False)

print(f"新的CSV文件已保存为 '{new_csv}'，包含了聚类结果。")


# # # 打印每个簇包含的样本数量以验证
# # for j in range(k):
# #     count = np.sum(cluster_assignment == j)
# #     print(f"簇 {j}: {count} 个样本")

# # # # 可选：查看聚类结果
# # # # 例如，打印前5个簇的样本索引
# # for j in range(k):
# #     samples = np.where(cluster_assignment == j)[0]
# #     print(f"簇 {j} 的样本索引: {samples}")
