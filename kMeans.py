import tensorflow as tf
import numpy as np
from  genData import Data
import plotter as plt


def buildGraph(alpha, k,dim):
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(421)

    trainingInput = tf.placeholder(tf.float32, shape=(None, dim)) # Placeholder for input
    centroid = tf.get_variable('mean', shape=(k,dim), initializer=tf.initializers.random_normal()) # Initialize cluster centres
    distanceSquared = distanceFunc(trainingInput,centroid) # Call distance funtion
    loss = tf.math.reduce_sum(tf.math.reduce_min(distanceSquared,0)) # Calculate loss
    optimizer =tf.train.AdamOptimizer(learning_rate= alpha).minimize(loss) # Adam Optimizer
    return optimizer, loss,  distanceSquared, centroid, trainingInput
    
def kMeans():
    ### Parameters you can vary ###
    alpha = 1e-3 # Learning Rate
    features = 3 # Number of features you want to use
    epochs = 800 # Number of Epochs
    k = 2 # Number of clusers
    a = Data(False,features) # Change to True if you want to find the top X features. If set to False, this will use the top 3 features that are set as SV%, EVGF, and EVGA 
    ### End of Parameter you can vary ###

    
    trainData, trainTarget, validData, validTarget, testData = a.featureExtractor() # Condense number of columns to top X features
    numTrain = trainData.shape[0] # Find number of points in training set
    numValid = validData.shape[0] # Find number of points in validation set
  
   # Initialize containers for plotting
    trainLoss = np.full((epochs, 1), np.inf)
    validLoss = np.full((epochs, 1), np.inf)

    optimizer, loss,  distanceSquared, centroid, trainingInput = buildGraph(alpha,k,features)
    init = tf.global_variables_initializer()   # Initialize session
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,epochs):
            _, trainLoss[i], dist, mU = sess.run([optimizer, loss,  distanceSquared, centroid], feed_dict = {trainingInput:trainData}) # Optimize to find new cluster centre
            validLoss[i],distV = sess.run([loss,  distanceSquared], feed_dict = {trainingInput: validData}) # Feed validation data to calculate loss

        [distT] = sess.run([distanceSquared], feed_dict = {trainingInput: testData}) # Calcuate distance of points test set

        plt.kMeansP(trainLoss, validLoss, numTrain,numValid, epochs,k) # Plot

        # Assign training, validation and test data points to clusters
        assign = np.argmin(dist,0) 
        assignV = np.argmin(distV,0)
        assignT = np.argmin(distT,0)

        # Plot scatter plots
        plt.scatter(trainData, assign, mU, trainTarget)      
        plt.scatter(validData, assignV, mU,validTarget)
        plt.scatter(testData, assignT, mU, testData.index) # Send index to show team name on plot
               
        print("\nTraining Set Accuracy is: ", np.mean(trainTarget == assign))
        print("Validation Set Accuracy is: ", np.mean(validTarget == assignV))
        return assign, assignV, assignT



def distanceFunc(X, mu): # Calculate distance squared of each point to each cluster
    expandPoints = tf.expand_dims(X, 0)
    expandCentroid = tf.expand_dims(mu, 1)
    return tf.reduce_sum(tf.square(tf.subtract(expandPoints, expandCentroid)), 2)



if __name__ == "__main__":
    assign, assignV, assignT= kMeans()