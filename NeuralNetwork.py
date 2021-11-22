from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

df = pd.read_csv(r'sign_mnist_train.csv')
df_x = df.iloc[:, 1:].values
df_y = df.iloc[:, 0].values

df_test = pd.read_csv(r'sign_mnist_test.csv')
df_test_x = df_test.iloc[:, 1:].values
df_test_y = df_test.iloc[:, 0].values

#train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)

                                                    #nodes, layers
clf = MLPClassifier(alpha = 0.05, hidden_layer_sizes = (80, 70), activation = 'relu', random_state = 1, max_iter = 300, learning_rate_init = 0.05)
clf.fit(df_x, df_y)

#print(clf.predict(df_test_x[0:3, :]))

# Plot the LEARNING CURVE
plt.title("Evolution of training error during training")
plt.xlabel("Iterations (epochs)")
plt.ylabel("Training error")
plt.plot(clf.loss_curve_)
plt.show()

# Evaluate acuracy on TEST data
score = clf.score(df_test_x[0:100, :],df_test_y[0:100])
print("Acuracy (on test set) = ", score)