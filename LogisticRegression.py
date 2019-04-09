import tensorflow as tf
import numpy as np
from genData import Data
import plotter as plt

def buildGraph(dim, alpha):
    tf.set_random_seed(421)
    X = tf.placeholder(tf.float32, [None, dim])
    yTarget = tf.placeholder(tf.float32, [None,1])
    reg = tf.placeholder(tf.float32,None, name='regularizer')
    weights = tf.Variable(tf.truncated_normal(shape=[dim,1], stddev=0.5), name='weights')
    b = tf.Variable(tf.truncated_normal(shape=[1,1], stddev=0.5), name='biases')


    model = tf.matmul(X,weights) + b

    CE = tf.nn.sigmoid_cross_entropy_with_logits(logits = model,labels = yTarget) + reg*tf.nn.l2_loss(weights) 
    loss = tf.reduce_mean(CE)
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss) 

    yPredicted = tf.round(tf.nn.sigmoid(model))
    correct = tf.cast(tf.equal(yPredicted, yTarget), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)*100
    return  X, yTarget, weights, b, loss, optimizer, accuracy, reg, tf.nn.sigmoid(model)

def Logistic():
   dim = 3
   alpha=1e-3
   regularization = 0.1
   batch_size = 13
   epochs = 200

   a = Data(True, dim)
   trainData, trainTarget, validData, validTarget, testData = a.featureExtractor()
   

   trainLoss = np.full((epochs, 1), np.inf)
   validLoss = np.full((epochs, 1), np.inf)
   trainAcc = np.zeros((epochs,1))
   validAcc = np.zeros((epochs,1))

   X, Y_target, W, b, lossfn, optimizer, accuracy, reg, yPred = buildGraph(dim, alpha)   # Building the graph
   init = tf.global_variables_initializer()   # Initialize session
   with tf.Session() as sess:
      sess.run(init)
      batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

      for i in range(epochs): # Loop across epochs
         Y_split = np.split(trainTarget,batch_number) # Split into the number of batches
         X_split = np.split(trainData,batch_number) # Split into the number of batches

         for j in range(len(X_split)): # Loop through each batch
            _,trainLoss[i], trainAcc[i] = sess.run([optimizer,lossfn, accuracy], feed_dict = {X: X_split[j], Y_target: np.expand_dims(Y_split[j],1), reg: regularization}) # Let us OPTIMIZE!

         validLoss[i], validAcc[i] = sess.run([lossfn,accuracy], feed_dict = {X: validData, Y_target: np.expand_dims(validTarget,1), reg: regularization}) # Store validation error and accuracy

      Final_Weight, Final_Bias = sess.run([W,b]) # Stores Final Weights and Biases in a form that is not a Tensor Object
      predict = sess.run([yPred], feed_dict = {X:testData})

      plt.logNet(trainLoss,validLoss,trainAcc,validAcc,epochs)

      print("\nTraining Set Accuracy is: ", trainAcc[-1])
      print("Validation Set Accuracy is: ", validAcc[-1])

      sess.close()

   return Final_Weight, Final_Bias, testData, predict


    
if __name__ == "__main__":
   Final_Weight, Final_Bias, testData, predict = Logistic()
   predict=np.reshape(np.asarray(predict),(-1,1))
   plt.postProcess(testData,1-predict)