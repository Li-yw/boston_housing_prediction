"""
数据加载模块
功能：加载数据、查看基本信息、划分训练测试集
"""
import pandas as pd  # 数据分析库，用于处理表格数据
import numpy as np  # 数值计算库，提供数组和矩阵运算
from sklearn.datasets import fetch_california_housing  # 从sklearn加载加州房价数据集
from sklearn.model_selection import train_test_split  # 用于将数据划分为训练集和测试集
import warnings  # 警告控制模块

warnings.filterwarnings('ignore')  # 忽略所有警告信息，避免输出干扰


def load_data():
    """
    加载房价数据集
    注意：由于原波士顿数据集被移除，我们使用加州房价数据集
    """
    # 加载加州房价数据集
    housing = fetch_california_housing()  # 返回Bunch对象，包含data(特征矩阵)、target(目标值)、feature_names(特征名)

    # 转换为DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)  # housing.data: 特征矩阵(20640×8); housing.feature_names: 特征列名列表
    df['PRICE'] = housing.target  # housing.target: 目标变量（房价中位数，单位：万美元）
    df.to_csv("../data/output_utf8.csv", encoding="utf-8")
    print("=" * 50)  # 打印50个"="作为分隔线
    print("数据集基本信息:")
    print(f"数据集形状: {df.shape}")  # df.shape: 返回(行数, 列数)的元组
    print(f"特征数量: {len(housing.feature_names)}")  # housing.feature_names: 8个特征名
    print(f"样本数量: {len(df)}")  # len(df): 数据集的总行数(样本数)
    print("=" * 50)

    return df, housing.feature_names  # 返回DataFrame和特征名列表


def explore_data(df):
    """
    数据探索函数
    """
    print("\n📊 数据探索:")
    print("1. 前5行数据:")
    print(df.head())  # df.head(): 默认返回前5行数据，可传参数如head(10)返回前10行

    print("\n2. 数据统计信息:")
    print(df.describe())  # df.describe(): 输出数值列的统计摘要(计数、均值、标准差、最小值、四分位数、最大值)

    print("\n3. 数据信息:")
    print(df.info())  # df.info(): 输出DataFrame的概况(列名、非空数量、数据类型、内存占用)

    print("\n4. 缺失值检查:")
    print(df.isnull().sum())  # df.isnull(): 返回布尔矩阵标记空值; .sum(): 按列统计缺失值数量

    return df  # 返回原DataFrame（探索不修改数据）


def split_data(df, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    参数:
        df: 包含特征和目标列的DataFrame
        test_size: 测试集占比，默认0.2（即20%作为测试集）
        random_state: 随机种子，默认42，保证每次划分结果一致（可复现）
    """
    X = df.drop('PRICE', axis=1)  # 删除PRICE列，axis=1表示按列删除，X为特征矩阵
    y = df['PRICE']  # 提取PRICE列作为目标变量y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )  # train_test_split参数: X-特征, y-目标, test_size-测试集比例, random_state-随机种子
       # 返回: X_train-训练特征, X_test-测试特征, y_train-训练目标, y_test-测试目标

    print("\n📈 数据划分结果:")
    print(f"训练集大小: {X_train.shape}")  # X_train.shape: 训练集的(样本数, 特征数)
    print(f"测试集大小: {X_test.shape}")  # X_test.shape: 测试集的(样本数, 特征数)

    return X_train, X_test, y_train, y_test  # 返回划分后的四个数据集


if __name__ == "__main__":  # 当直接运行该脚本时执行（被导入时不执行）
    # 测试数据加载
    df, features = load_data()  # df: 包含特征的DataFrame; features: 特征名列表
    explore_data(df)  # 对数据集进行探索性分析
