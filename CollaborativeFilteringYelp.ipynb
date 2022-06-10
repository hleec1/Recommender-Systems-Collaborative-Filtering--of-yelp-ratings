import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from collections import Counter
from google.colab import files

uploaded = files.upload()
train = pd.read_json(io.BytesIO(uploaded['yelp_academic_dataset_review.json']), lines=True)

# Get the 100 most common users row and then get the most common 50 business ratings out of the users
users = []
biz = []
common_users = []
common_biz = []
users = train['user_id'].tolist()
common_users = Counter(users).most_common(100)
for i in range(100):
  x, _ = common_users[i]
  temp = (train[train['user_id'] == x])
  biz.extend(temp['business_id'].tolist())
common_biz = Counter(biz).most_common(50)

# Create the rating matrix from the data above.
rat_mat = np.zeros((100,50))
for i in range(len(common_users)):
  for k in range(len(common_biz)):
    x, _ = common_users[i]
    temp = train[train['user_id'] == x]
    y, _ = common_biz[k]
    temp2 = temp[temp['business_id'] == y]
    if temp2.empty == True:
      rat_mat[i,k] = 0
    else:
      rat_mat[i,k] = (temp2['stars'].iloc[0])
      
print(rat_mat)

# Function to split train and test data, we need to make sure by doing the leave-one-out split that we do not leave all empty columns or rows. 
#This function verifies that that condition is not met during splitting.
def train_test_split(rat_mat):
    f = 0
    test = np.zeros(rat_mat.shape)
    training = rat_mat.copy()
    counter = 0
    while f != -1:
      seed1 = np.random.choice(100, 1)
      seed2 = np.random.choice(50, 1)
      if rat_mat[seed1[0], seed2[0]] == 0:
        continue
      elif rat_mat[seed1[0], seed2[0]] != 0:
        temporal = training[seed1[0], seed2[0]]
        training[seed1[0], seed2[0]] = 0
        transrat = training.T
        if np.all(training[seed1[0]] == training[seed1[0]][0]):
          training[seed1[0], seed2[0]] == temporal
          continue
        elif np.all(transrat[seed2[0]] == transrat[seed2[0]][0]):
          training[seed1[0], seed2[0]] == temporal
          continue
        else:
          training[seed1[0], seed2[0]] = 0
          test[seed1[0], seed2[0]] = rat_mat[seed1[0], seed2[0]]
          counter += 1
      if counter == 109:
        f = -1
    return training, test

training, test = train_test_split(rat_mat)
# Matrix factorization class for Collaborative filtering
class MF():
    def __init__(self, R, K, alpha, beta, epochs):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
            ]
        training_process = []
        for i in range(self.epochs):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
        print("Final error: %.4f" % mse)
        return training_process

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        
# This function is to get the Mean-Squared-Error of the Actual vs Predicted values in the matrix        
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.square(np.subtract(actual, pred)).mean()
    
# Training and loss graph     
mf = MF(training, K=20, alpha=0.05, beta=0.002, epochs=300)
lossgraph = mf.train()
plt.title("MSE with rank 20")
plt.plot(*zip(*lossgraph4))
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
# Test prediction vs actual
print("MSE of test data: " + str(get_mse(mf4.full_matrix(), test)))
