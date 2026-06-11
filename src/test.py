from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
print(iris.feature_names)