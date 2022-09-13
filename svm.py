import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier
import seaborn as sns

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
data = pd.read_csv("iris.data", names=columns)

x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

sns.pairplot(data, hue='Class_labels')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = SVC(kernel="rbf")

predicted_model = model.predict(x_test)
