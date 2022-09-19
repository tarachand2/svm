# https://www.analyticsvidhya.com/blog/2022/06/iris-flowers-classification-using-machine-learning/
# https://data-flair.training/blogs/iris-flower-classification/

import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support vecor classifier
import seaborn as sns
import matplotlib.pyplot as plt
# columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
data = pandas.read_csv("C:/Users/khorw/ML_project/iris.csv")
print(data)

x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

sns.pairplot(data, hue="Class_labels")
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(len(x_train))
print(len(x_test))
model = SVC(kernel="rbf")

model.fit(x_train, y_train)

predicted_model = model.predict(x_test)

print(model.score(x_test, y_test))
