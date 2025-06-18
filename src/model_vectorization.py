import numpy as np
from tqdm import tqdm
##################### Distance functions #####################
def Euclidean_distance(x_test,x_train):
    """Use Euclidean_distance with vectorization 
    Distance=sqrt(sum(x_test-x_train)^2)=sqrt(x_test^2+x_train^2-2x_test*x_train)"""
    distance=np.sqrt(np.sum(x_test**2,axis=1).reshape(-1, 1)+np.sum(x_train**2,axis=1).reshape(1, -1) 
                                -2*np.dot(x_test,x_train.T))
    return distance

def Manhattan_distance(x_test,x_train):
    """Manhattan_distance function is not optimized like the distance calculation function with vectorization, 
    however it can be extended to create and call C++ files to optimize calculation speed."""
    distance= np.sum(np.abs(x_test[:, np.newaxis, :].astype(np.float16) - x_train[np.newaxis, :, :].astype(np.float16)).astype(np.float16), axis=2).astype(np.float16)
    return distance

def Minkowski_distance(x_test,x_train,p):
    """Minkowski_distance function is not optimized like the distance calculation function with vectorization, 
    however it can be extended to create and call C++ files to optimize calculation speed."""
    distance = np,pow(np.sum(np.abs(x_test[:, np.newaxis, :].astype(np.float16)  - x_train[np.newaxis, :, :].astype(np.float16) ) ** p, axis=2),1 / p).astype(np.float16) 
    return distance

##################### K_nearest_Neighbors Vectorization #####################
class K_nearest_Neighbors_Vectoarization:
    def __init__(self,k_n_neighbors,distance_func="Minkowski",p=2):
        self.p=p
        distance_map={"Minkowski":lambda a,b :Minkowski_distance(a,b,self.p),
                      "Euclidean":Euclidean_distance,
                      "Manhattan":Manhattan_distance}
        self.distance_func=distance_map.get(distance_func,Minkowski_distance)
        self.k_n_neighbors=k_n_neighbors
        # fit data    
    def fit(self,x,y):
        if len(y.shape)==1:
            y=np.array(y).reshape(-1,1)
        self.x_train=x
        self.y_train=y
    
    def _predict_vectorization(self,x_test):

        Distance_matrix=self.distance_func(x_test,self.x_train)
        k_indexs=np.argsort(Distance_matrix,axis=1)[:,0:self.k_n_neighbors]     #get k indexs arrange from min to max 
        K_nearest_labels=self.y_train[k_indexs]     #y data retrieval with k indexs and row

        """ar=np.array([[9,2,3,4,5,4,3,2,9,1,7,1,9],
                        [1,2,1,5,8,1,6,9,7,2,3,0,7]]) 
        np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=ar) = [9 1]"""
        y_pred = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=K_nearest_labels)
        return y_pred
    
    def predict_vectorization(self,x_array,batch_size=100):
        """ if push all x_test to model can over RAM
            need split data into batchs to put into the model"""
        y_pred_array=[]
        for batch in tqdm(range(0,x_array.shape[0],batch_size), desc="Predict: ", unit="sample"):
            y_pred_array.append(self._predict_vectorization(x_array[batch:batch+batch_size]))      #put batch_size samples to model

        return np.array(y_pred_array).reshape(-1)
    
    def accuracy_vectorization(self,y_true,y_pred):
        return np.mean(y_pred==y_true)
