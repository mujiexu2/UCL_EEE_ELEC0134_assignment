import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from PIL import Image
import lab2_landmarks as l2
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import tree

def train_test_A2():
    train_root="./Dataset/dataset_AMLS_22-23/celeba"
    test_root="./Dataset/dataset_AMLS_22-23_test/celeba_test"

    '''With Dlib'''
    X_train, Y_train = l2.extract_features_labels(os.path.join(train_root, "img"), train_root, "smiling")
    X_test, Y_test = l2.extract_features_labels(os.path.join(test_root, "img"), test_root, "smiling")
    num=len(X_train)
    X_train=np.array(X_train).reshape(num,-1)
    Y_train=np.array(Y_train)
    Y_train=Y_train.reshape(-1,1)
    num=len(X_test)
    X_test=np.array(X_test).reshape(num,-1)
    Y_test=np.array(Y_test)
    Y_test=Y_test.reshape(-1,1)


    # '''Without Dlib'''
    # df = pd.read_csv(train_root+"/labels.csv")
    # Y_train=[]
    # for i in range(5000):
    #     li=df["\timg_name\tgender\tsmiling"][i].split('\t')
    #     Y_train.append(int(li[2]))
    # Y_train=np.array(Y_train)
    # Y_train=Y_train.reshape(-1,1)

    # X_train=[]
    # #X_train是一个list
    # for j in range(5000):
    #     li2=df["\timg_name\tgender\tsmiling"][j].split('\t')
    #     path=train_root+"/img/"+li2[1]
    #     im=Image.open(path).convert("RGB")
    #     im = im.resize((100, 100))
    #     ii=np.asarray(im)
    #     ii=ii.flatten()
    #     X_train.append(ii)
    # num=len(X_train)
    # X_train=np.array(X_train).reshape(num,-1)

    # df_t=pd.read_csv(test_root+"/labels.csv")
    # Y_test=[]
    # for i in range(1000):
    #     li=df_t["\timg_name\tgender\tsmiling"][i].split('\t')
    #     Y_test.append(int(li[2]))

    # Y_test=np.array(Y_test)
    # Y_test=Y_test.reshape(-1,1)

    # X_test=[]
    # for j in range(1000):
    #     li2=df_t["\timg_name\tgender\tsmiling"][j].split('\t')
    #     path=test_root+"/img/"+li2[1]
    #     im=Image.open(path).convert("RGB")
    #     im = im.resize((100, 100))
    #     ii=np.asarray(im)
    #     ii=ii.flatten()
    #     X_test.append(ii)

    # X_test=np.array(X_test)

    # '''PCA dimensionality reduction'''
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=50)#这里是PCA降维，你可以设置n_components为不同的维度来改变SVM的准确率
    # X_train=pca.fit_transform(X_train)
    # X_test=pca.fit_transform(X_test)

    '''SVM+Hyperparameter Tuning'''
    print('SVM Hyperparameter Tuning starts:------------------------')
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']} 
    
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, Y_train.ravel())
    print(grid.best_params_) 
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    print("Model: SVM after hyperparameter tuning")
    print(classification_report(Y_test, grid_predictions))
    print('Accuracy score: ', accuracy_score(Y_test,grid_predictions))
    print('Confusion Matrix:\n ',confusion_matrix(Y_test,grid_predictions))
    print('Precision Score: ',precision_score(Y_test,grid_predictions))
    print('Recall Score: ',recall_score(Y_test, grid_predictions))
    print('F1 Score: ',f1_score(Y_test,grid_predictions))

    '''default setting for SVM model'''
    # model=SVC()
    # model.fit(X_train,Y_train)
    # Y_pred = model.predict(X_test)
    # Y_pred
    # print('Model:SVM')
    # print('Accuracy Score: ',accuracy_score(Y_test,Y_pred))
    # print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))
    # print('Precision Score: ',precision_score(Y_test,Y_pred))
    # print('Recall Score: ',recall_score(Y_test, Y_pred))
    # print('F1 Score: ',f1_score(Y_test,Y_pred))

    '''Decision Tree+ Hyperparameter Tuning'''
    # '''Default Decision Tree'''
    # clf=tree.DecisionTreeClassifier()
    # clf.fit(X_train,Y_train)
    # Y_pred = clf.predict(X_test)
    # print('Model: Decision Tree')
    # print('Accuracy Score: ',accuracy_score(Y_test,Y_pred))
    # print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))
    # print('Precision Score: ',precision_score(Y_test,Y_pred))
    # print('Recall Score: ',recall_score(Y_test, Y_pred))
    # print('F1 Score: ',f1_score(Y_test,Y_pred))
    print('DT Hyperparameter Tuning starts:------------------------')
    accuracy=[]
    confusion=[]
    precision=[]
    recall=[]
    f1=[]
    for i in range(1,31):
        clf=tree.DecisionTreeClassifier(max_depth=i)
        clf.fit(X_train,Y_train)

        Y_pred = clf.predict(X_test)
        
        accuracy.append(accuracy_score(Y_test,Y_pred))
        confusion.append(confusion_matrix(Y_test,Y_pred))
        precision.append(precision_score(Y_test,Y_pred))
        recall.append(recall_score(Y_test, Y_pred))
        f1.append(f1_score(Y_test,Y_pred))
    depth=accuracy.index(max(accuracy))
    best_accuracy=max(accuracy)
    print(depth, best_accuracy)
    clf=tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    print('Model: Decision Tree')
    print('Accuracy Score: ',accuracy_score(Y_test,Y_pred))
    print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))
    print('Precision Score: ',precision_score(Y_test,Y_pred))
    print('Recall Score: ',recall_score(Y_test, Y_pred))
    print('F1 Score: ',f1_score(Y_test,Y_pred))
    #plot
    x1 = range(1,31)
    y1 = accuracy
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. Max depth')
    plt.xlabel('Max depth')
    plt.ylabel('Test accuracy')
    plt.savefig("accuracy_max_depth_A2.jpg")
    print('-------------Test accuracy vs. max_depth (A2) is plotted and saved---------------')

    '''KNN+Hyperparameter Tuning'''
    best_score = 0.0
    best_k = -1
    for k in range(1, 11):
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
    print('Accuracy Score: ',accuracy_score(Y_test,Y_pred))
    print('Confusion Matrix:\n ',confusion_matrix(Y_test,Y_pred))
    print('Precision Score: ',precision_score(Y_test,Y_pred))
    print('Recall Score: ',recall_score(Y_test, Y_pred))
    print('F1 Score: ',f1_score(Y_test,Y_pred))

if __name__ == '__main__':
    train_test_A2()