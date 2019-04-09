import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns



class Data:
    def __init__(self, run=False, maxFeat=3):
        self.run = run
        self.maxFeat = maxFeat
        if (run == False) & (maxFeat !=3):
            raise ValueError("You shall not pass! Cannot select number of features without making run equal to True")


    def importData(self):
        removeIndices = ["Rk","GP", "W", "L", "OL", "PTS", "PTS%","SOW","SOL","SRS","GA","GF", "SOS"]
        combinedData = pd.read_csv("The PLAYOFFS - Data.csv")
        combinedData.iloc[:,1] = combinedData.iloc[:,1].map(lambda x: x.strip('*'))
        combinedData = combinedData.set_index("Team")
        combinedData = combinedData.drop(removeIndices,axis=1)
        combinedData = combinedData.sample(frac=1, random_state=421)
        unFilteredtrainData = combinedData.iloc[:,:-1]
        trainTargets = combinedData.iloc[:,-1]-1
    
        unFilteredtestData = pd.read_csv("The PLAYOFFS - 2018.csv")
        unFilteredtestData.iloc[:,1] = unFilteredtestData.iloc[:,1].map(lambda x: x.strip('*'))
        unFilteredtestData = unFilteredtestData.set_index("Team")
        unFilteredtestData = unFilteredtestData.drop(removeIndices,axis=1)
        unFilteredtestData = unFilteredtestData.sample(frac=1,  random_state=421)
       
        return unFilteredtrainData, trainTargets, unFilteredtestData


    def normalizer(self,X):
        return (X - X.mean(axis=0))/X.std(axis=0)
    
    
    def convertOneHot(self,target):  
        oneTarget = np.zeros((target.shape[0], 2))
        oneTarget[np.arange(target.shape[0]), target] = 1
        return oneTarget
    

    def featureExtractor(self):
        unFilteredtrainData, trainTargets, unFilteredtestData = self.importData()
        np.random.seed(421)
        xTrain, xValid, yTrain, yValid = train_test_split(unFilteredtrainData, trainTargets, test_size=0.2) # 80% training and 20% test

        if self.run:
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
    
            sns.barplot(x=maxImportance, y=maxImportance.index)
            plt.title("Which Features Are Important?", fontsize = 32)
            plt.xlabel('Importance Scale', fontsize = 28)
            plt.ylabel('Features', fontsize = 28)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.show()
                    
            trainData = xTrain.filter(maxImportance[0:self.maxFeat].index)
            validData = xValid.filter(maxImportance[0:self.maxFeat].index)
            testData = unFilteredtestData.filter(maxImportance[0:self.maxFeat].index)

        
        else:

            trainData = xTrain.filter(['SV%','EVGF','EVGA'])
            validData = xValid.filter(['SV%','EVGF','EVGA'])
            testData = unFilteredtestData.filter(['SV%', 'EVGF','EVGA'])
       
        trainData = self.normalizer(trainData)  
        validData = self.normalizer(validData)
        testData = self.normalizer(testData)


        return trainData, yTrain, validData, yValid, testData


    
            
