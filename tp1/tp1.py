import numpy as np
import matplotlib.pyplot as plt

#pluie, arrosoir, watson, holmes

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print ("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))
prob_arrosoir = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print ("Pr(Arrosoir)={}\n".format(np.squeeze(prob_arrosoir)))
watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print ("Pr(G1|Pluie)={}\n".format(np.squeeze(watson)))
# # Regression Logistique


from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data

y = digits.target
y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK


X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 500
minibatch_size = len(y) // 20

losses = []
losses_validation = []
accuracies = []


def softmax(x):
    pass


def get_accuracy(X, y, W):
    pass

def get_grads(y, y_pred, X):
    pass


for epoch in range(nb_epochs):
    loss = 0
    accuracy = 0
    for i in range(0, X_train.shape[0], minibatch_size):
        pass
    epoch += 1
    losses_validation.append(loss_val)
    accuracies.append(accuracy)
    if accuracy > best_accuracy:
        pass
    
accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
print(accuracy_on_unseen_data)
plt.plot(losses)

plt.imshow(best_W[4, :].reshape(8,8))


