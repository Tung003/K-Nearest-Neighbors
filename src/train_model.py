import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.metrics import f1_score, precision_score,recall_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from read_data import split_data
from model import K_nearest_Neighbors

def train(x_train,y_train):
    model=K_nearest_Neighbors(k_n_neighbors=5,distance_func="Minkowski")
    model.fit(x_train,y_train)
    return model

def main(): 
    x_train,x_test,y_train,y_test=split_data()
    ################ Scaler data #################
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test =scaler.transform(x_test)

    x_test=x_test[0:100]
    y_test=y_test[0:100]
    start=time.time()
    ################# Model train ################
    model=train(x_train,y_train)
    ################# Model test #################
    y_test_pred=model.predict(x_test)
    ############## Model accuracy ################
    accuracy=model.accuracy(y_test_pred,y_test)
    print("model accuracy: ",accuracy)
    ############# Model evaluation ###############

    f1_scores=f1_score(y_test,y_test_pred,average='macro')
    recall_scores=recall_score(y_test,y_test_pred,average='macro')
    precision_scores=precision_score(y_test,y_test_pred,average='macro')
    print("F1_score  = ","%.3f"%f1_scores)
    print("Recall    = ","%.3f"%recall_scores)
    print("Precision = ","%.3f"%precision_scores)

    print("-----------------Classification report-----------------\n",classification_report(y_test,y_test_pred))
    #confusion_matrix
    confusion_matrix_result = confusion_matrix(y_test, y_test_pred)
    print("confusion_matrix \n",confusion_matrix_result)

    #plot confusion_matrix
    df_cm = pd.DataFrame(confusion_matrix_result)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("/home/chu-tung/Desktop/machine_learning/Logistic_Regression/outputs/metrics_refined_f1.png")
    plt.show()
    ################## Time run ##################
    end_time=time.time()
    print(f"\ntotal time run: {end_time-start:.3f} s")
    ##############################################

if __name__=="__main__":
    main()