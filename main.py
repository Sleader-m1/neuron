import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
 

eps = 0.001 #Коэффициент скорости обучения 

alphabet =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) *(1 - sigmoid(x))

def activationFunction(x):
    return sigmoid(x)

def activationFunctionDerivative(x):
    return sigmoidDerivative(x)


neuron_count = 20

W = np.random.randint(1, 10, (neuron_count, 35)) / 10
bias = np.random.randint(1, 10, (20, 1)) / 10

W_second = np.random.randint(1, 10, (26, neuron_count)) / 10
bias_second =  np.random.randint(1, 10, (26, 1)) / 10

def getCalculation(input_v, W_ = -1, bias_ = -1, W_second_ = -1, bias_second_ = -1):
    W_ = W_ if type(W_) != int else W
    bias_ = bias_ if type(bias_) != int else bias
    bias_second_ = bias_second_ if type(bias_second_) != int else bias_second
    W_second_ = W_second_ if type(W_second_) != int else W_second
    
    ext_input = np.append(input_v, [1])
    H_raw = np.dot( np.column_stack((W_,bias_)), ext_input) 
    H_vector = np.array([activationFunction(H) for H in H_raw])
    O_raw = np.dot(np.column_stack((W_second_,bias_second_)), np.append(H_vector, [1]))
    O_vector=  np.array([activationFunction(O) for O in O_raw])

    return O_vector

def getMistakeCoef(answer_v, output_v):
    result = 0
    for i in range(26):
        result += (answer_v[i]-output_v[i])**2
    return result

def gradientW(answer, x_input):
    result_W = W.copy()
    d = eps
    for i in range(len(W)):
        for j in range(len(W[i])):
            new_W = W.copy()
            new_W[i][j] += d
            first_f = getMistakeCoef(answer, getCalculation(x_input, W_ = new_W))
            new_W[i][j] -= 2*d
            second_f = getMistakeCoef(answer, getCalculation(x_input, W_ = new_W))
            result_W[i][j] = (first_f-second_f)*eps/(2*d)
    return result_W

def gradientWSecond(answer, x_input):
    result_W = W_second.copy()
    d = eps
    for i in range(len(W_second)):
        for j in range(len(W_second[i])):
            new_W = W_second.copy()
            new_W[i][j] += d
            first_f = getMistakeCoef(answer, getCalculation(x_input, W_second_ = new_W))
            new_W[i][j] -= 2*d
            second_f = getMistakeCoef(answer, getCalculation(x_input, W_second_ = new_W))
            result_W[i][j] =  (first_f-second_f)*eps/(2*d)        
    return result_W

def gradientBias(answer, x_input):
    result_B = bias.copy()
    d = eps
    for i in range(len(bias)):
        new_B = bias.copy()
        new_B[i] += d
        first_f = getMistakeCoef(answer, getCalculation(x_input, bias_= new_B))
        new_B[i] -= 2*d
        second_f = getMistakeCoef(answer, getCalculation(x_input, bias_=new_B))
        result_B[i] = (first_f-second_f)*eps/(2*d)        
    return result_B

def gradientBiasSecond(answer, x_input):
    result_B = bias_second.copy()
    d = eps
    for i in range(len(bias_second)):
        new_B = bias_second.copy()
        new_B[i] += d
        first_f = getMistakeCoef(answer, getCalculation(x_input, bias_second_= new_B))
        new_B[i] -= 2*d
        second_f = getMistakeCoef(answer, getCalculation(x_input, bias_second_=new_B))
        result_B[i] = (first_f-second_f)*eps/(2*d)        
    return result_B

 
def learn(input_v, answer):
    global W, W_second, bias, bias_second
    
    W -= gradientW(answer, input_v) 
    bias -= gradientBias(answer, input_v) 
    W_second -= gradientWSecond(answer, input_v) 
    bias_second = gradientBiasSecond(answer, input_v)


def startLearning():
    directories = ['greyscale/' + i + '/' for i in alphabet]
    for j in range(1000):
        print(f'{j+1}/1000 started')
        for i in range(len(directories)):
            image = Image.open(f'{directories[i]}{alphabet[i]}1.png')
            image_array = np.asarray(image).tolist()
            res = []
            E = np.ones(35)
            for imag in image_array:
                res = np.concatenate((res, imag))
            input_v = E - res / 255
            answer = np.zeros(26)
            answer[i] = 1
            learn(input_v, answer)
    
    weights = {
            'W1': W.tolist(),
            'W2': W_second.tolist(),
            'B': bias.tolist(), 
            'B2': bias_second.tolist()
        }

    with open('weights_100.json', 'w') as outfile:
        json.dump(weights, outfile, indent=4)

    print("Learning finiched")


def getResult(file):
    global W, W_second, bias, bias_second
    W = json.load(open("weights_100.json", "r"))['W1']
    W_second = json.load(open("weights_100.json", "r"))['W2']
    bias = json.load(open("weights_100.json", "r"))['B']
    bias_second = json.load(open("weights_100.json", "r"))['B2']

    image = Image.open(file)
    image_array = np.asarray(image).tolist()
    res = []
    E = np.ones(35)
    for imag in image_array:
        res = np.concatenate((res, imag))
    input_v = E - res / 255
    
    out = getCalculation(input_v)
    fig, ax = plt.subplots()

    x = np.array([chr(i) for i in range(65, 91)])
    y = np.array(out)
    ax.bar(x, y)
    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(12)  # высота Figure
    plt.show()
    return alphabet[np.argmax(out)]





startLearning()
print(getResult('greyscale/a/a1.png'))
print(getResult('greyscale/b/b1.png'))
