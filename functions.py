import numpy as np
import pandas as pd


#random generator
rg = np.random.default_rng()

#FIRST DEFINITIONS OF ALL NEEDED FUNCTIONS
##function to generate ramdon data, based on custom features
def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    ##print(features)
    weights = rg.random((1, n_values))[0] #this is necesary because return a bidimensional array
    targets = np.random.choice([0,1], n_features)
    ##print(targets)
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["targets"] = targets
    
    return data, weights

def example_data():
    features = [[0,0,1],[1,1,1],[1,0,1],[0,1,1]] #lenght of dataset is 4, with 3 dimensions or layers
    ##print(features)
    weights = rg.random((1, 3))[0] #this is necesary because return a bidimensional array
    targets = [0,1,1,0] #it has to be the same number of lenght of dataset
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"]) ##lenght of dataset =4
    data["targets"] = targets
    print("\n initial data: \n", data)
    print("\n initial weights \n", weights)
    return data, weights

def excel_data():
    df_excel = pd.read_excel('data.xls')
    df_excel_columnas = df_excel.columns
    print("Excel Heads: \n", df_excel_columnas)
    df_dataset_lenght = df_excel.iloc[0,0]
    print("\n Dataset lenght: ",df_dataset_lenght)
    variables = int(input("\n How many variables for inputs exist? \n"))
    features=[]
    columns_input=[]
    for i in range(df_dataset_lenght):
        intermedia=[]
        columns_input=[]
        for j in range(variables):
            intermedia.append(df_excel.iloc[i,j+2])
            columns_input.append("x"+str(j+1))
        features.append(intermedia)
    print("\n inputs: \n", columns_input,features)
    variables_targets = int(input("\n How many variables for outputs exist? \n"))
    targets=[]
    for i in range(df_dataset_lenght):
        intermedia=[]
        for j in range(variables_targets):
            targets.append(df_excel.iloc[i,j+2+variables])
        ##targets.append(intermedia)
    print("\n outputs: \n", targets) 
    data = pd.DataFrame(features, columns=columns_input)
    data["targets"] = targets
    weights = rg.random((1, variables))[0]
    print("\n initial data: \n", data)
    print("\n initial weights \n", weights)
    return data, weights

#basically the sum of a line ecuation, where line is y = mx + b
def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

#function cross_entropy to calculate error, between [0,1]
def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))

def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)

def evaluation_neuronal(example, bias, weights):
    sum_weights = get_weighted_sum(example, weights, bias)
    ##print("\nresult sum_weights: ", sum_weights)
    return sigmoid(sum_weights)
    
def save_csv(data, weights, bias, l_rate, epochs, epoch_loss, loss, average_loss):
    print("\n Weights: \n",weights)
    df= pd.DataFrame()
    #df["Data"] = data
    df["Weights"]=weights
    #df["epochs"]=epochs
    #df["epoch_loss"]=epoch_loss
    return df.to_csv('results.csv', index=False, decimal=",")
    
    
    
#HERE FINISH DEFINITIONS OF FUNCTIONS
    
