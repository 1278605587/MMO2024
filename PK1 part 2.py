import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# 生成一个简单的数据集
np.random.seed(0)
data = {
    'feature1': np.random.randint(1, 100, 100),
    'feature2': np.random.randint(1, 100, 100),
    'feature3': np.random.rand(100) * 100,
    'target': np.random.randint(0, 2, 100)  # 假设有一个二分类目标变量
}
df = pd.DataFrame(data)

# 执行特征选择

# 目标变量
target_variable = 'target'

# 选择最佳特征的数量
k_best_features = 2

# 从数据中删除目标变量
X = df.drop(columns=[target_variable])

# 确定目标变量
y = df[target_variable]

# 使用互信息方法进行特征选择
selector = SelectKBest(score_func=mutual_info_regression, k=k_best_features)
selected_features = selector.fit_transform(X, y)

# 获取选定特征的索引
selected_indices = selector.get_support(indices=True)

# 获取选定特征的名称
selected_feature_names = X.columns[selected_indices]

# 打印选定特征
print("选定的特征:")
print(selected_feature_names)
