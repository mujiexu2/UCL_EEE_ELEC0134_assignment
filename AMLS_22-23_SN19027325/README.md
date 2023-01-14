# UCL-ELEC0134 assignment
UCL ELEC0134 course final assignment by Mujie Xu 19027325

This readme gives the structure of the code and the procedures about how to train and test the models.


### Task A1&A2 used Models: SVM, Decision Tree, K-nearest Neighbour
### Task B1&B2 used Models: SVM, Random Forest, K-nearest Neighbour

The computer I used to do the project is MacBook Pro (13-inch, M1, 2020). Version is 12.6.

### Python Libraries used:
- numpy
- sklearn
- pandas
- pillow
- os
- Tensorflow


These are commonly used libraries, and it is easy to install. However, my computer is a M1-chip Mac, this gives many limitations on installing libraries. 

- Tensorflow libraries should be installed with Miniforge, and the python version should be 3.9.
    - This is the link teaches how to install TensorFlow:
        Chinese version:
            1. https://blog.csdn.net/crist_meng/article/details/121947662?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121947662-blog-115792419.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121947662-blog-115792419.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1
            2. https://zhuanlan.zhihu.com/p/474212619
        English Version
            1. https://claytonpilat.medium.com/tutorial-tensorflow-on-an-m1-mac-using-jupyter-notebooks-and-miniforge-dbb0ef67bf90

- When running l2.extract_feature_labels(), 
    if you are using a windows computer,
    the code :file_name= img_path.split('/')[-1] needs to be modified into file_name= img_path.split('\')[-1] 

### Program Structure 
-- AMLS_22-23_SN19027325

&emsp; -- A1

&emsp;&emsp; -- A1.py

&emsp; -- A2

&emsp;&emsp;  A2.py

&emsp; -- B1

&emsp;&emsp; -- B1.py

&emsp; -- B2

&emsp;&emsp; -- B2.py

&emsp; -- main.py

### Program run instruction
You can run "python A1/A2/B1/B2.py" to run each task. And it can also be done with "python main.py" which 4 tasks will be run one by one.

A1/A2/B1/B2.py task included the code does hyperparameter tuning and the one with default setting(has no inputs). 
For A1 and A2, 
    SVM is with hyperparameter tuning function.
    DT is with hyperparameter tuning function.
    KNN is with hyperparameter tuning function.

For B1 and B2,
    SVM goes with the default setting, no inputs. model=SVC().This may give an unmatched accuracy with the report-written accuracy.
    RF is with hyperparameter tuning function.
    KNN is with hyperparameter tuning function.


    


