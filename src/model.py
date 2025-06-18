import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score,recall_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from read_data import split_data
import pandas as pd
import seaborn as sns
import time

############################# Distance Function ############################
def Minkowski_distance(a,b,p):
    return np.pow(np.sum(np.abs(np.pow(a-b,p))),(1 / p))

def Euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b) ** 2))

def Manhattan_distance(a,b):
    return np.sum(np.abs(a-b))


##################### K_nearest_Neighbors Class Normal #####################
class K_nearest_Neighbors:
    def __init__(self,k_n_neighbors,distance_func="Minkowski",p=2):
        self.k_n_neighbors=k_n_neighbors
        distance_map={"Minkowski":lambda a,b :Minkowski_distance(a,b,p),
                      "Euclidean":Euclidean_distance,
                      "Manhattan":Manhattan_distance}
        
        self.distance_func= distance_map.get(distance_func,Minkowski_distance)
        self.p=p

    def fit(self,x,y):
        if len(y.shape)==1:
            y=np.array(y).reshape(-1,1)
        self.x_train=x
        self.y_train=y

##################### point predict compute distance with for #####################
    def _predict(self,x):
        distance=[]
        for x_train in self.x_train:
            distance.append(self.distance_func(x,x_train))  #add distance in array
        k_indexs=np.argsort(distance)[0:self.k_n_neighbors] #get k indexs arrange from min to max 
        K_nearest_labels=self.y_train[k_indexs]   #y data retrieval with k indexs
        
        """np.bincount (EX: ar=np.array([1,2,3,4,5,4,3,2,1,1,7,1,9]) 
        np.bincount(ar)like:[0 4 2 2 2 1 0 1 0 1]
                            [0 1 2 3 4 5 6 7 8 9] => 1 appeared 4 times, 4 appeared 2 times
        np.argmax Returns the indices of the maximum values along an axis => return = 1"""

        y_pred=np.argmax(np.bincount(K_nearest_labels.reshape(-1))) 
        return y_pred
    
    def predict(self,x_array):
        y_pred_array=[]
        for x in tqdm(x_array, desc="Đang dự đoán", unit="mẫu"):
            y_pred_array.append(self._predict(x))
        return np.array(y_pred_array).reshape(-1)

    def accuracy(self,y_pred,y_true):
        return np.mean(y_pred==y_true)
################## End class##################
