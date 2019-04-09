import tensorflow as tf
import numpy as np
from genData import Data
import plotter as plt
from sklearn.utils import shuffle


def buildGraph(dim, alpha):
   tf.reset_default_graph() # Clear any previous junk
   tf.set_random_seed(421)
   
   X = tf.placeholder(tf.float32, [None, dim], name='inputs') # Placeholder for input
   yTarget = tf.placeholder(tf.float32, [None,1], name= 'targets') # Placeholder for targets
   reg = tf.placeholder(tf.float32,None, name='regulaizer') # Regularization Placeholder
   weights = tf.Variable(tf.truncated_normal(shape=[dim,1], stddev=0.5), name='weights') # Initialize weights
   b = tf.Variable(tf.truncated_normal(shape=[1,1], stddev=0.5), name='biases') # Initialize bias


   model = tf.matmul(X,weights) + b # Define model

   CE = tf.nn.sigmoid_cross_entropy_with_logits(logits = model,labels = yTarget) + reg*tf.nn.l2_loss(weights) # define Cross entropy
   loss = tf.reduce_mean(CE) # Define loss
   optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss) # Use Adam optimizer

   yPredicted = tf.round(tf.nn.sigmoid(model)) # Predict by rounding to 0 or 1
   correct = tf.cast(tf.equal(yPredicted, yTarget), dtype=tf.float32) # Find number of correct predictions made compared to the labels
   accuracy = tf.reduce_mean(correct)*100 # Find average accuracy
   return  X, yTarget, weights, b, loss, optimizer, accuracy, reg, tf.nn.sigmoid(model)

def Logistic():
   ### Parameters you can vary ###
   alpha = 1e-3 # Learning Rate
   batch_size = 13 # Batch size. Number must be a factor of training set size
   features = 3 # Number of features you want to use
   epochs = 200 # Number of Epochs  
   regularization = 0.0 # Regularizastion term
   a = Data(False,features) # Change to True if you want to find the top X features. If set to False, this will use the top 3 features that are set as SV%, EVGF, and EVGA 
   ### End of Parameter you can vary ###


   trainData, trainTarget, validData, validTarget, testData = a.featureExtractor() # Condense number of columns to top X features
   

   # Initialize containers for plotting
   trainLoss = np.full((epochs, 1), np.inf) 
   validLoss = np.full((epochs, 1), np.inf)
   trainAcc = np.zeros((epochs,1))
   validAcc = np.zeros((epochs,1))


   X, Y_target, W, b, lossfn, optimizer, accuracy, reg, yPred = buildGraph(features, alpha)   # Building the graph
   init = tf.global_variables_initializer()   # Initialize session
   with tf.Session() as sess:
      sess.run(init)
      batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

      for i in range(epochs): # Loop across epochs
         trainData, trainTarget = shuffle(trainData,trainTarget) # Shuffle training data each epoch

         X_split = np.split(trainData,batch_number) # Split into the number of batches
         Y_split = np.split(trainTarget,batch_number) # Split into the number of batches

         for j in range(len(X_split)): # Loop through each batch
            _,trainLoss[i], trainAcc[i] = sess.run([optimizer,lossfn, accuracy], feed_dict = {X: X_split[j], Y_target: np.expand_dims(Y_split[j],1), reg: regularization}) # Let us OPTIMIZE!

         validLoss[i], validAcc[i] = sess.run([lossfn,accuracy], feed_dict = {X: validData, Y_target: np.expand_dims(validTarget,1), reg: regularization}) # Store validation error and accuracy

      Final_Weight, Final_Bias = sess.run([W,b]) # Stores Final Weights and Biases in a form that is not a Tensor Object
      plt.logNet(trainLoss,validLoss,trainAcc,validAcc,epochs)

      print("\nTraining Set Accuracy is: ", trainAcc[-1])
      print("Validation Set Accuracy is: ", validAcc[-1])

      predict = sess.run([yPred], feed_dict = {X:testData}) # Predict for this year

      sess.close()

   return Final_Weight, Final_Bias, testData, predict


    
if __name__ == "__main__":
   Final_Weight, Final_Bias, testData, predict = Logistic()
   predict=np.reshape(np.asarray(predict),(-1,1))
   plt.postProcess(testData,1-predict)