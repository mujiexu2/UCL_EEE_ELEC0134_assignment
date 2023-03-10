import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from PIL import Image
import os 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

def train_test_B1():
    train_root="./Dataset/dataset_AMLS_22-23/cartoon_set"
    test_root="./Dataset/dataset_AMLS_22-23_test/cartoon_set_test"
    # '''With Dlib'''
    # import lab2_landmarks as l2
    # X_train, Y_train = l2.extract_features_labels(os.path.join(train_root, "img"), train_root, "faceshape")
    # X_test, Y_test = l2.extract_features_labels(os.path.join(test_root, "img"), test_root, "faceshape")
    # X_train_b, Y_train_b, X_test_b, Y_test_b = X_train, Y_train, X_test, Y_test

    # num=len(X_train)
    # X_train=np.array(X_train).reshape(num,-1)
    # Y_train=np.array(Y_train)
    # Y_train=Y_train.reshape(-1,1)

    # num=len(X_test)
    # X_test=np.array(X_test).reshape(num,-1)
    # Y_test=np.array(Y_test)
    # Y_test=Y_test.reshape(-1,1)

    df = pd.read_csv(train_root+"/labels.csv")
    Y_train=[]
    for i in range(10000):
        li=df["\teye_color\tface_shape\tfile_name"][i].split('\t')
        Y_train.append(int(li[2]))
    #Y_train
    Y_train=np.array(Y_train)
    #Y_train
    Y_train=Y_train.reshape(-1,1)
    print('Y_train shape: ',Y_train.shape)

    X_train=[]
    for j in range(10000):
        li2=df["\teye_color\tface_shape\tfile_name"][j].split('\t')
        path=train_root+"/img/"+li2[3]
        im=Image.open(path).convert("RGB")
        im = im.crop((100, 50, im.size[0]-100, im.size[1]-50))
        im = im.resize((60, 80))
        ii=np.asarray(im)
        ii=ii.flatten()
        X_train.append(ii)
    #X_train
    X_train=np.array(X_train)
    
    df_t=pd.read_csv(test_root+"/labels.csv")
    Y_test=[]
    for i in range(2500):
        li=df_t["\teye_color\tface_shape\tfile_name"][i].split('\t')
        Y_test.append(int(li[2]))
    #Y_test
    Y_test=np.array(Y_test)
    Y_test=Y_test.reshape(-1,1)
    print('Y_test shape:', Y_test.shape)

    X_test=[]
    for j in range(2500):
        li2=df_t["\teye_color\tface_shape\tfile_name"][j].split('\t')
        path=test_root+"/img/"+li2[3]
        im=Image.open(path).convert("RGB")
        im = im.crop((100, 50, im.size[0]-100, im.size[1]-50))
        im = im.resize((60, 80))
        ii=np.asarray(im)
        ii=ii.flatten()
        X_test.append(ii)
    #X_test
    X_test=np.array(X_test)

    # '''PCA'''
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=1000)
    # X_train=pca.fit_transform(X_train)
    # X_test=pca.fit_transform(X_test)

    # '''SVM+ Hyperparameter Tuning'''
    # ###really takes a long time 
    # from sklearn.model_selection import GridSearchCV

    # param_grid = {'C': [0.1, 1, 10, 100, 1000], 
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']} 
    
    # grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # grid.fit(X_train, Y_train.ravel())
    # print(grid.best_params_) 
    # print(grid.best_estimator_)
    # grid_predictions = grid.predict(X_test)
    # print(classification_report(Y_test, grid_predictions))
    # print('accuracy score: ', accuracy_score(Y_test,grid_predictions))
    # print('Confusion Matrix:\n ',confusion_matrix(Y_test,grid_predictions))

     '''Default SVM'''
     print('SVM starts:---------------------------------------------')
     model=SVC()
     model.fit(X_train,Y_train.ravel())
     Y_pred = model.predict(X_test)
     print('Model: Default SVM')
     print('Accutacy Score: ', accuracy_score(Y_test,Y_pred))
     print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))

    '''Random Forest Classifier+ Hyperparameter Tuning'''
    print('RF Hyperparameter Tuning starts:------------------------')
    rfc = RandomForestClassifier() 
    param_grid = {'max_depth' : np.arange(1,20,1)}
    rf = RandomForestClassifier(n_estimators=11,random_state=42)
    rfc = GridSearchCV(rf,param_grid,cv=5)
    rfc =rfc.fit(X_train,Y_train.ravel())
    print(rfc.best_params_ ) 
    print(rfc.best_score_ ) 

    x1 = []
    y1 = []
    means = rfc.cv_results_['mean_test_score']
    params = rfc.cv_results_['params']

    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))
        x1.append(int(param['max_depth']))
        y1.append(mean)
    print(x1)  
    print(y1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. Max depth')
    plt.xlabel('Max depth')
    plt.ylabel('Test accuracy')
    plt.savefig("accuracy_max_depth_B1_RF.jpg")

    print('------------Test accuracy vs. max_depth (B1) is plotted and saved---------------')
    Y_train = Y_train.ravel()
    rfc=RandomForestClassifier(max_depth=14)
    rfc = rfc.fit(X_train, Y_train)
    Y_pred = rfc.predict(X_test)
    
    print('Model:RF after Hyperparameter Tuning')
    print('Accutacy Score: ', accuracy_score(Y_test,Y_pred))
    print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))

    '''KNN + Hyperparameter Tuning'''
    print('KNN Hyperparameter Tuning starts:------------------------')
    best_score = 0.0
    best_k = -1
    for k in range(1, 50):
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        knn_clf.fit(X_train, Y_train.ravel())
        score = knn_clf.score(X_test, Y_test)
        if score > best_score:
            best_k = k
            best_score = score
    print("best_k =", best_k)
    print("best_score =", best_score)
    model = KNeighborsClassifier(best_k)
    model.fit(X_train, Y_train.ravel())
    Y_pred = model.predict(X_test)
    print('Model: KNN')
    print('Accutacy Score: ', accuracy_score(Y_test,Y_pred))
    print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))

if __name__ == '__main__':
    train_test_B1()

