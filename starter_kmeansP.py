import tensorflow as tf
import numpy as np
from  genData import Data
import plotter as plt


def buildGraph(alpha, k,dim):
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(421)

    trainingInput = tf.placeholder(tf.float32, shape=(None, dim))
    centroid = tf.get_variable('mean', shape=(k,dim), initializer=tf.initializers.random_normal())
    distanceSquared = distanceFunc(trainingInput,centroid)
    loss = tf.math.reduce_sum(tf.math.reduce_min(distanceSquared,0))
    optimizer =tf.train.AdamOptimizer(learning_rate= alpha, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
    return optimizer, loss,  distanceSquared, centroid, trainingInput
    
def kMeans():
    dim = 3
    a = Data(False,dim)
    trainData, trainTarget, validData, validTarget, testData = a.featureExtractor()
    numTrain = trainData.shape[0]
    numValid = validData.shape[0]
  
    alpha = 1e-3
    k = 2
    epochs = 800  

    trainLoss = np.full((epochs, 1), np.inf)
    validLoss = np.full((epochs, 1), np.inf)

    optimizer, loss,  distanceSquared, centroid, trainingInput = buildGraph(alpha,k,dim)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,epochs):
            _, trainLoss[i], dist, mU = sess.run([optimizer, loss,  distanceSquared, centroid], feed_dict = {trainingInput:trainData})
            validLoss[i],distV = sess.run([loss,  distanceSquared], feed_dict = {trainingInput: validData})

        [distT] = sess.run([distanceSquared], feed_dict = {trainingInput: testData})

        plt.kMeansP(trainLoss, validLoss, numTrain,numValid, epochs,k)

        assign = np.argmin(dist,0)
        assignV = np.argmin(distV,0)
        assignT = np.argmin(distT,0)

        plt.scatter(trainData, assign, mU, trainTarget)      
        plt.scatter(validData, assignV, mU,validTarget)
        plt.scatter(testData, assignT, mU, testData.index)
               
        print("\nTraining Set Accuracy is: ", np.mean(trainTarget == assign))
        print("Validation Set Accuracy is: ", np.mean(validTarget == assignV))
        return assign, assignV, assignT



def distanceFunc(X, mu):
    expandPoints = tf.expand_dims(X, 0)
    expandCentroid = tf.expand_dims(mu, 1)
    return tf.reduce_sum(tf.square(tf.subtract(expandPoints, expandCentroid)), 2)



if __name__ == "__main__":
    assign, assignV, assignT= kMeans()