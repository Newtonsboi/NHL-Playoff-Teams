import tensorflow as tf
import numpy as np
import plotter as plt
from genData import Data


def buildGraph(dim, alpha,dropProb, hiddenUnits):
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(421)

    labels = tf.placeholder(shape=(None, 2), dtype='int32')
    reg = tf.placeholder(tf.float32,None, name='regulaizer')
    isTraining = tf.placeholder(tf.bool)

    weights = {
    'w1': tf.get_variable('W1', shape=(dim,hiddenUnits), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'w2': tf.get_variable('W2', shape=(hiddenUnits,hiddenUnits), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'w3': tf.get_variable('W3', shape=(hiddenUnits,2), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),

}
    biases = {
    'b1': tf.get_variable('B1', shape=(hiddenUnits), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'b2': tf.get_variable('B2', shape=(hiddenUnits), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),
    'b3': tf.get_variable('B3', shape=(2), initializer=tf.contrib.layers.xavier_initializer(uniform = False)),

}
    X = tf.placeholder(tf.float32, shape=(None, dim))
    layer1 = tf.nn.bias_add(tf.matmul(X, weights['w1']), biases['b1'])

    x1 = tf.nn.relu(layer1)
    layer2 = tf.nn.bias_add(tf.matmul(x1, weights['w2']), biases['b2'])


    toReLU = tf.cond(isTraining, lambda: tf.nn.dropout(layer2, rate = dropProb), lambda: layer2)
    x2 = tf.nn.relu(toReLU)

    layer3 = tf.nn.bias_add(tf.matmul(x2, weights['w3']), biases['b3'])
    predict = tf.nn.softmax(layer3)

    outputClass = tf.argmax(predict, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict)) +reg*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2'])+tf.nn.l2_loss(weights['w3']))

    correct_prediction = tf.equal(outputClass, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100

    optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
    return optimizer, loss, X, labels, reg, accuracy, predict, isTraining

def CNN():
    dim = 4
    a = Data(False,dim)
    trainData, trainTarget, validData, validTarget, testData = a.featureExtractor()
    trainTarget = a.convertOneHot(trainTarget)
    validTarget = a.convertOneHot(validTarget)

    alpha = 1e-4
    dropProb = 0.5
    batch_size = 13
    epochs = 150 
    hiddenUnits = 500
    regularization = 0.01

    trainLoss = np.full((epochs, 1), np.inf)
    validLoss = np.full((epochs, 1), np.inf)
    trainAcc = np.zeros((epochs,1))
    validAcc = np.zeros((epochs,1))

    optimizer, loss, X, labels, reg, accuracy, predict, isTraining = buildGraph(dim,alpha, dropProb, hiddenUnits)
    init = tf.global_variables_initializer()   
    with tf.Session() as sess:
        sess.run(init)
        batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

        for i in range(epochs): # Loop across epochs
            Y_split = np.split(trainTarget,batch_number) # Split into the number of batches
            X_split = np.split(trainData,batch_number) # Split into the number of batches

            for j in range(len(X_split)): # Loop through each batch
           # Let us OPTIMIZE! Set isTraining to True to enable dropout for training only
                _, trainLoss[i], trainAcc[i] = sess.run([optimizer,loss,accuracy], feed_dict = {X: X_split[j], labels: Y_split[j], reg: regularization, isTraining:True})

            validLoss[i], validAcc[i]= sess.run([loss,accuracy], feed_dict = {X: validData, labels: validTarget, reg: regularization, isTraining:False})


        predictor = sess.run([predict], feed_dict = {X: testData,isTraining: False})
        plt.logNet(trainLoss,validLoss,trainAcc,validAcc,epochs)
        print("\nTraining Set Accuracy is: ", trainAcc[-1])
        print("Validation Set Accuracy is: ", validAcc[-1])

       
        sess.close()
        
    return testData, predictor

   
if __name__ == "__main__":
    testData, predictor = CNN()
    predictor=np.reshape(np.asarray(predictor),(-1,2))
    plt.postProcess(testData,predictor)