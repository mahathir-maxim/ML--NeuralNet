# -*- coding: utf-8 -*-
"""NeuralNet.ipynb

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv("https://cs4375data.s3.amazonaws.com/iris.csv")

        
    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        dataset=self.raw_input
        dataset.isnull().sum()
        correlation= dataset.corr().round(2)
        print("correlation: ")
        print(correlation)
        sns.heatmap(data=correlation, annot=True)
        self.processed_data= dataset

        return 0

    
    
    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        iris = self.raw_input
        iris.columns = ["Sep_Length", "Sep_Width", "Pet_Length", "Pet_Width", "Class"]
        df = pd.DataFrame(iris, columns=iris.columns)
        # print(df.head())
        # print(iris.head())

        
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        meanVal = np.mean(X, axis = 0)  
        stdDev = np.std(X, axis= 0, ddof = 1)  
        X_updated= X - meanVal
        X_normalized = X_updated/stdDev
        X= X_normalized
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state=5)

        
        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        Methods = ['MLP_1', 'MLP_2','MLP_3', 'MLP_4','MLP_5', 'MLP_6','MLP_7', 
                   'MLP_8', 'MLP_9','MLP_10', 'MLP_11','MLP_12', 'MLP_13','MLP_14',
                   'MLP_15', 'MLP_16','MLP_17', 'MLP_18','MLP_19', 'MLP_20','MLP_21',
                   'MLP_22', 'MLP_23','MLP_24']
        Metrics = ['Activation', 'Learning Rate', 'Max Iterations', 
                   'Num Hidden Layers', 'Train Acc',
                   'Test Acc', 'Train Err', 'Test Err']
        compare_df = pd.DataFrame(index = Methods, columns = Metrics)
        acc_val = []
        iter_val = []
        acc_val_tot = []
        iter_val_tot = []
        

        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train)
        # Now apply the transformations to the data:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        methods_int = 0
        
        for i in range(0, len(activations)):
          for j in range(0, len(learning_rate)):
            for k in range(0, len(max_iterations)):
              for x in range(0, len(num_hidden_layers)):
                  #using  activation = logistic,
                  #       learning rate = 0.01,
                  #       max iterations = 100,
                  #       hidden layers = 2,
                  mlp = MLPClassifier(activation=activations[i], 
                                      learning_rate_init=learning_rate[j],
                                      max_iter=max_iterations[k],
                                      hidden_layer_sizes=(num_hidden_layers[x]),
                                      random_state=5)
                  mlp.fit(X_train,y_train)
                  test_predictions = mlp.predict(X_test) #getting y predictions on test x vals
                  train_predictions = mlp.predict(X_train) #getting y predictions on train x vals

                  y_test_int = pd.get_dummies(y_test).values.argmax(1)
                  test_predictions_int = pd.get_dummies(test_predictions).values.argmax(1)
                  y_train_int = pd.get_dummies(y_train).values.argmax(1)
                  train_predictions_int = pd.get_dummies(train_predictions).values.argmax(1)

                  acc_score_test = accuracy_score(y_test_int,test_predictions_int) 
                  acc_score_train = accuracy_score(y_train_int,train_predictions_int)

                  Test_MSE = mean_squared_error(y_test_int, test_predictions_int)
                  Train_MSE= mean_squared_error(y_train_int, train_predictions_int)

                  train_RMSE = math.sqrt(Train_MSE)
                  test_RMSE= math.sqrt(Test_MSE)

                  # print("Train: \n")
                  # print(acc_score_train)
                  # print(train_RMSE)

                  # print("Test: \n")
                  # print(acc_score_test)
                  # print(test_RMSE)

                  conf_matrix = confusion_matrix(y_test,test_predictions)
                  # print(confusion_matrix(y_test,test_predictions))

                  # print(classification_report(y_test,predictions))
                  compare_df.loc[Methods[methods_int], ['Activation']] = activations[i]
                  compare_df.loc[Methods[methods_int], ['Learning Rate']] = learning_rate[j]
                  compare_df.loc[Methods[methods_int], ['Max Iterations']] = max_iterations[k]
                  compare_df.loc[Methods[methods_int], ['Num Hidden Layers']] = num_hidden_layers[x]
                  compare_df.loc[Methods[methods_int], ['Test Acc']] = acc_score_test
                  compare_df.loc[Methods[methods_int], ['Test Err']] = test_RMSE
                  compare_df.loc[Methods[methods_int], ['Train Acc']] = acc_score_train
                  compare_df.loc[Methods[methods_int], ['Train Err']] = train_RMSE
                  
                  if max_iterations[k] == 100:
                    acc_val.append(acc_score_test)
                    iter_val.append(100)
                    acc_val_tot.append(acc_score_test)
                    iter_val_tot.append(100)
                  else:
                    acc_val.append(acc_score_test)
                    iter_val.append(200)
                    acc_val_tot.append(acc_score_test)
                    iter_val_tot.append(200)

                  methods_int = methods_int + 1
                  print("activations: " + str(activations[i]))
                  print("learning_rate: " + str(learning_rate[j]))
                  print("max_iterations: " + str(max_iterations[k]))
                  print("num_hidden_layers: " + str(num_hidden_layers[x]))
                  plt.plot(mlp.loss_curve_)
                  plt.title("Loss/Accuracy History")
                  plt.xlabel('Max Iterations')
                  plt.ylabel('Loss Value')
                  plt.show()

                  # Plot the model history for each model in a single plot
                  # model history is a plot of accuracy vs number of epochs
                  # you may want to create a large sized plot to show multiple lines
                  # in a same figure.
          print("activations: " + activations[i])
          plt.plot(iter_val, acc_val, 'ro')
          plt.title("Accuracy of " + str(activations[i]))
          plt.xlabel('Max Iterations')
          plt.ylabel('Accuracy Value')
          plt.show()
          acc_val = []
          iter_val = []

        plt.plot(iter_val_tot, acc_val_tot, 'ro')
        print("Accuracy of all activations and their iterations: ")
        plt.title("Accuracy of activations")
        plt.xlabel('Max Iterations')
        plt.ylabel('Accuracy Value')
        plt.show()
        print("Result Table: ")
        print(compare_df)      
        return 0

if __name__ == "__main__":
    neural_network = NeuralNet("https://cs4375data.s3.amazonaws.com/iris.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()

