import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



def kMeansP(trainLoss, validLoss, numTrain,numValid,epochs,k=2): # Plotting function for kMeans
    plt.figure(1)
    plt.clf()
    plt.plot(trainLoss/numTrain, 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(validLoss/numValid, 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.title("K = %i Loss vs Epoch" % k, fontsize = 32)

    plt.ylabel("Loss Value", fontsize = 30)
    plt.xlabel("Epoch", fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlim((0,epochs))
    plt.legend(ncol=1, fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()




def scatter(X, cluster, mU, n): # Scatter function for kMeans
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:,2], c= cluster, s=20, cmap='RdBu')
    plt.title("Scatter Plot of Normalized Parameters", fontsize= 32)
    ax.set_xlabel(X.columns[0], fontsize = 20)
    ax.set_ylabel(X.columns[1], fontsize = 20)
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(X.columns[2],  rotation = 0, fontsize = 20)
    
    for i in range(len(X)): #plot each point and its index as text above. Only for test data
        ax.text(X.iloc[i,0],X.iloc[i,1],X.iloc[i,2],  '%s' % (str(n[i])), size=13, zorder=1,  
            color='k')
    plt.show()

def logNet(trainLoss,validLoss,trainAcc,validAcc,epochs): # Plotting function for both Logistic and Neural Net 
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(trainLoss, 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(validLoss, 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.title('Loss Plot', fontsize = 32)
    plt.xlim(0,epochs)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(212)
    plt.plot(trainAcc, 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(validAcc, 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.title('Accuracy Plot', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,epochs)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(50,100)
    plt.grid(which = 'both', axis = 'both')
    plt.show()


def postProcess(testData, winner): # First round match-up predictor for Neural Net and Logistic
    testData["Playoff Chances"] = winner[:,0]
    Atlantic = testData.filter(['Tampa Bay Lightning','Boston Bruins', 'Toronto Maple Leafs','Montreal Canadiens','Ottawa Senators', 'Florida Panthers','Buffalo Sabres', 'Detroit Red Wings'],axis=0)
    Atlantic = Atlantic.sort_values("Playoff Chances",ascending=False)

    Metro = testData.filter(['Washington Capitals', 'Pittsburgh Penguins','New York Islanders', 'New York Rangers', 'Columbus Blue Jackets', 'Carolina Hurricanes', 'New Jersey Devils','Philadelphia Flyers'],axis=0)  
    Metro = Metro.sort_values("Playoff Chances",ascending=False)

    wildCardEast = pd.concat([Atlantic.iloc[3:],Metro.iloc[3:]])
    wildCardEast = wildCardEast.sort_values("Playoff Chances", ascending=False)
    
    if  Atlantic.iloc[0,-1] > Metro.iloc[0,-1]:
        print("\nEastern Playoff Prediction:\n", Atlantic.index[0], "vs.", wildCardEast.index[1])
        print("\n", Atlantic.index[1], "vs.", Atlantic.index[2] )
        print("\n", Metro.index[0], "vs.", wildCardEast.index[0] )
        print("\n", Metro.index[1], "vs.", Metro.index[2] )

    else:
        print("\nEastern Playoff Prediction:\n", Metro.index[0], "vs.", wildCardEast.index[1] )
        print("\n", Metro.index[1], "vs.", Metro.index[2] )
        print("\n", Atlantic.index[0], "vs.", wildCardEast.index[0])
        print("\n", Atlantic.index[1], "vs.", Atlantic.index[2] )

    

    Central = testData.filter(['Dallas Stars','Colorado Avalanche', 'Chicago Blackhawks','St. Louis Blues','Winnipeg Jets', 'Nashville Predators','Minnesota Wild'],axis=0)
    Central = Central.sort_values("Playoff Chances",ascending=False)

    Pacific = testData.filter(['Calgary Flames', 'Vancouver Canucks','Edmonton Oilers', 'Arizona Coyotes', 'Vegas Golden Knights', 'San Jose Sharks', 'Anaheim Ducks','Los Angeles Kings'],axis=0)  
    Pacific = Pacific.sort_values("Playoff Chances",ascending=False)

    wildCardWest = pd.concat([Central.iloc[3:],Pacific.iloc[3:]])
    wildCardWest = wildCardWest.sort_values("Playoff Chances", ascending=False)
    
    if  Central.iloc[0,-1] > Pacific.iloc[0,-1]:
        print("\nWestern Playoff Prediction:\n", Central.index[0], "vs.", wildCardWest.index[1])
        print("\n", Central.index[1], "vs.", Central.index[2] ) 
        print("\n", Pacific.index[0], "vs.", wildCardWest.index[0] )
        print("\n", Pacific.index[1], "vs.", Pacific.index[2] )

    else:
        print("\nWestern Playoff Prediction:\n", Pacific.index[0], "vs.", wildCardWest.index[1] )
        print("\n", Pacific.index[1], "vs.", Pacific.index[2] )
        print("\n", Central.index[0], "vs.", wildCardWest.index[0])
        print("\n", Central.index[1], "vs.", Central.index[2] ) 