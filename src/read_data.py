import pandas as pd
import os

data_train_path="/home/chu-tung/Desktop/machine_learning/K_Nearest_Neighbors/data/raw/mnist_train.csv"
data_test_path ="/home/chu-tung/Desktop/machine_learning/K_Nearest_Neighbors/data/raw/mnist_test.csv"

def split_data():
    if not os.path.exists(path=data_train_path):
        raise FileNotFoundError(f"File không tồn tại: {data_train_path}")
    elif not os.path.exists(path=data_test_path):
        raise FileNotFoundError(f"File không tồn tại: {data_test_path }")
        
    df_train=pd.read_csv(data_train_path)
    df_test =pd.read_csv(data_test_path )

    target="label"
    x_train=df_train.drop(target,axis=1)
    x_test =df_test .drop(target,axis=1)

    y_train=df_train[target]
    y_test =df_test [target]
    return x_train,x_test,y_train,y_test
