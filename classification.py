from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train, X_train.shape)

model = SVC(kernel='rbf', gamma=0.001)
model.fit(X_train, y_train)

print('test accuracy')
print(model.score(X_test, y_test))
