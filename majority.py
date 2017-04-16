import numpy as np

############################################
# activation function and its differential #
############################################
def sigmoid(x):
    return 1./(1. + np.exp(-x))

def diff_sigmoid(x):
    return sigmoid(x)*(1. - sigmoid(x))

######################################
# loss function and its differential #
######################################
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def diff_mean_squared_error(y,t):
    return y-t

########
# main #
########
def majority():
    ########################
    # parameter definition #
    ########################
    trainingDataNum = 28 # the number of training datas
    testDataNum = 4 # the number of test datas
    inputNodeNum = 5 # the number of nodes of input layer
    hiddenNodeNum = 3 # the number of nodes of hidden layer
    outputNodeNum = 1 # the number of nodes of output layer
    learningRate = 1 # learning rate
    epochNum = 5000 # the number of epoch
    errorThreshold = 0.0001 # the number of epoch
    ##############
    # load datas #
    ##############
    training = np.loadtxt('majority_training.csv', delimiter = ',')
    print(training)
    label = np.loadtxt('majority_label.csv', delimiter = ',')
    print(label)
    test = np.loadtxt('majority_test.csv', delimiter = ',')
    print(test)
    ##########################################################
    # inilialize weights, wh and wo, randomly                #   
    # wh is the weight between input layer and hidden layer  #
    # wo is the weight between hidden layer and output layer #
    ##########################################################
    rng = np.random
    wh = rng.randn(inputNodeNum, hiddenNodeNum)
    wo = rng.randn(hiddenNodeNum, )
    ########################################################
    # inilialize biases bh and bo to zero                  #
    # bh is the bias between input layer and hidden layer  #
    # bo is the bias between hidden layer and output layer #
    ########################################################
    bh = np.zeros((hiddenNodeNum,), float)
    bo = np.zeros((outputNodeNum,), float)

    ######################
    # training procedure #
    ######################
    count = 0
    err = 1.
    while (count < epochNum) and (err > errorThreshold):
        count += 1
        err = 0.
        for i in range(trainingDataNum):
            h = sigmoid(np.dot(training[i],wh)+bh)
            o = sigmoid(np.dot(h,wo)+bo)
            L_over_f = diff_mean_squared_error(o,label[i]) 
            f_over_o = diff_sigmoid(np.dot(h,wo)+bo)
            L_over_o = np.dot(f_over_o, L_over_f)
            o_over_wo = h.T
            subwo = np.dot(o_over_wo, L_over_o)
            # update wo and bo
            wo -= learningRate*subwo
            bo -= learningRate*L_over_o

            f_over_h = diff_sigmoid(np.dot(training[i],wh)+bh)
            h_over_wh = training[i]
            o_over_h = np.dot(f_over_h, wo)
            L_over_h = np.dot(o_over_h, L_over_o)
            subwh = np.outer(training[i], L_over_h)
            wh -= learningRate*subwh
            bh -= learningRate*L_over_h

            ##################
            # re-calculation #
            ##################
            h = sigmoid(np.dot(training[i],wh)+bh)
            o = sigmoid(np.dot(h,wo)+bo)
            err += mean_squared_error(o,label[i])
            
            if (i==trainingDataNum-1):
                print(err)

    #########################
    # test data calculation #
    #########################
    for i in range(testDataNum):
        h = sigmoid(np.dot(test[i],wh)+bh)
        o = sigmoid(np.dot(h,wo)+bo)
        print(o)

if __name__ == '__main__':
    majority()
