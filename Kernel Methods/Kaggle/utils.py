import numpy as np

import matplotlib.pyplot as plt

def reconstructImage(data):

    rows = data.values
    
    nb, N = rows.shape
    red_channel = rows[:,:N//3]
    min_red = red_channel.min(1).reshape(-1,1)
    max_red = red_channel.max(1).reshape(-1,1)
    red_channel = (red_channel-min_red)/(max_red-min_red)

    green_channel = rows[:,N//3:2*N//3]
    min_green = green_channel.min(1).reshape(-1,1)
    max_green = green_channel.max(1).reshape(-1,1)
    green_channel = (green_channel-min_green)/(max_green-min_green)

    blue_channel = rows[:,2*N//3:]
    min_blue = blue_channel.min(1).reshape(-1,1)
    max_blue = blue_channel.max(1).reshape(-1,1)
    blue_channel = (blue_channel-min_blue)/(max_blue-min_blue)

    newdata = np.hstack((red_channel,green_channel,blue_channel))
    newdata = newdata.reshape(nb,3,32,32).transpose(0,2,3,1)
    return newdata

def plot_image(data, index):

    row = data.iloc[index].values
    N = row.shape[0]
    red_channel = row[:N//3].reshape(-1,1)
    min_value = red_channel.min()
    max_value = red_channel.max()
    red_channel = np.array(1*(red_channel - min_value)/(max_value-min_value))
    
    green_channel = row[N//3:2*N//3].reshape(-1,1)
    min_value = green_channel.min()
    max_value = green_channel.max()

    
    green_channel = np.array(1*(green_channel - min_value)/(max_value-min_value))


    blue_channel = row[2*N//3:].reshape(-1,1)
    min_value = blue_channel.min()
    max_value = blue_channel.max()
    blue_channel = np.array(1*(blue_channel - min_value)/(max_value-min_value))
    

    image = np.hstack((red_channel,green_channel,blue_channel))
    image = image.reshape(32,32,3)
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    
def create_Submissioncsv(y_test):

    f = open("result/Yte.csv", "w")
    f.write("Id,Prediction\n")
    for n in range(len(y_test)):
        f.write("{},{}\n".format(int(n+1),y_test[n]))
    f.close()
    
def scaler(X_train, X_val = None, X_test = None):
    
    mu_X = X_train.mean()
    sigma_X = X_train.std()
        
    return (X_train - mu_X)/sigma_X, (X_val - mu_X)/sigma_X, (X_test - mu_X)/sigma_X 