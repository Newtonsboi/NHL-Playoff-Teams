import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# This class takes in the unfiltered data, and reduces it for use by extracting the total number of features the user has specified

class Data:
    def __init__(self, run=False, maxFeat=3):
        self.run = run
        self.maxFeat = maxFeat
        if (run == False) & (maxFeat !=3): # Raises error in the case when we select the number of features being not 3 and the feature updater is off 
            raise ValueError("You shall not pass! Cannot select number of features without making run equal to True")


    def importData(self):
        removeIndices = ["Rk","GP", "W", "L", "OL", "PTS", "PTS%","SOW","SOL","SRS","GA","GF", "SOS"] # Define indices that influence ranking
        combinedData = pd.read_csv("The PLAYOFFS - Data.csv")  # Import data
        combinedData.iloc[:,1] = combinedData.iloc[:,1].map(lambda x: x.strip('*')) # Some datasets have a *, remove them
        combinedData = combinedData.set_index("Team") # Set the team as the index
        combinedData = combinedData.drop(removeIndices,axis=1) # Remove the indices
        combinedData = combinedData.sample(frac=1, random_state=421) # Shuffle the dataset
        unFilteredtrainData = combinedData.iloc[:,:-1] # Seperate the training data from the training targets
        trainTargets = combinedData.iloc[:,-1]-1
    
        unFilteredtestData = pd.read_csv("The PLAYOFFS - 2018.csv") # Import test set, which is this year's data
        unFilteredtestData.iloc[:,1] = unFilteredtestData.iloc[:,1].map(lambda x: x.strip('*'))
        unFilteredtestData = unFilteredtestData.set_index("Team")
        unFilteredtestData = unFilteredtestData.drop(removeIndices,axis=1)
        unFilteredtestData = unFilteredtestData.sample(frac=1,  random_state=421)
       
        return unFilteredtrainData, trainTargets, unFilteredtestData


    def normalizer(self,X): # Normalize columns
        return (X - X.mean(axis=0))/X.std(axis=0)
    
    
    def convertOneHot(self,target):  # Converts to OneHot Matrix
        oneTarget = np.zeros((target.shape[0], 2))
        oneTarget[np.arange(target.shape[0]), target] = 1
        return oneTarget
    

    def featureExtractor(self): # Selects the top X features
        unFilteredtrainData, trainTargets, unFilteredtestData = self.importData()
        np.random.seed(421)
        xTrain, xValid, yTrain, yValid = train_test_split(unFilteredtrainData, trainTargets, test_size=0.2) # Splits trianing set into 80% training and 20% test

        if self.run: # If we want to find the top X features, run this loop. Uses random forest classifier to find the most influential features
            maxAcc = 0
            for i in range(1000):
                clf = RandomForestClassifier(n_estimators=19, max_depth=3) 
                clf = clf.fit(xTrain,yTrain)
                featureImportance = pd.Series(clf.feature_importances_,index=list(xTrain)).sort_values(ascending=False) 
                yPred=clf.predict(xValid)
                accuracy = metrics.accuracy_score(yValid, yPred)
                if accuracy > maxAcc:
                    maxAcc = accuracy
                    maxImportance = featureImportance
    
            sns.barplot(x=maxImportance, y=maxImportance.index) # Visualize the important features
            plt.title("Which Features Are Important?", fontsize = 32)
            plt.xlabel('Importance Scale', fontsize = 28)
            plt.ylabel('Features', fontsize = 28)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.show()
                    
            # Narrow dataset to just the important features 
            trainData = xTrain.filter(maxImportance[0:self.maxFeat].index) 
            validData = xValid.filter(maxImportance[0:self.maxFeat].index)
            testData = unFilteredtestData.filter(maxImportance[0:self.maxFeat].index)

        
        else:
            # If feature run is not True, then just return pre-selected important features set below
            trainData = xTrain.filter(['SV%','EVGF','EVGA'])
            validData = xValid.filter(['SV%','EVGF','EVGA'])
            testData = unFilteredtestData.filter(['SV%', 'EVGF','EVGA'])
       
       # Normalize the data before returning
        trainData = self.normalizer(trainData)  
        validData = self.normalizer(validData)
        testData = self.normalizer(testData)


        return trainData, yTrain, validData, yValid, testData