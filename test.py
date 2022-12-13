import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt

eps = 0.01 #Коэффициент скорости обучения 

alphabet =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) *(1 - sigmoid(x))

def tanh(x):
    a = np.zeros(len(x))
    for i in range(len(a)):
            a[i] = ((np.exp(x[i]) - np.exp(-x[i]))) / (np.exp(x[i]) + np.exp(-x[i]))
    return a

def tanhDerivative(x):
    a = np.zeros(len(x))
    for i in range(len(a)):
            a[i] = 4 / ((np.exp(x[i]) + np.exp(-x[i]))**2)
    return a

def activationFunction(x):
    return sigmoid(x)

def activationFunctionDerivative(x):
    return sigmoidDerivative(x)

class Neuron:
    def __init__(self):
        self.W = np.random.randint(1, 10, (20, 35)) / 10

        self.W_second = np.random.randint(1, 10, (26, 20)) / 10
    
    def getCalculation(self, input):
        #умножение матриц и векторов делать через np.dot()
        H_raw = np.dot(self.W, input)
        H_vector = np.array([activationFunction(H) for H in H_raw])

        O_raw = np.dot(self.W_second, H_vector)
        O = activationFunction(O_raw)

        return O, H_vector

    def Learn(self, vector, answer):
        lmbd = 0.01
        N = 450000
        count = len(vector)
        i = 0
        for i in range(N):
            index = np.random.randint(0, count)
            if i % 1000 == 0:
                print(i)
            x = vector[index]
            x_true = answer[index]
            y, out = self.getCalculation(x)
            e = y - x_true
            delta = e * activationFunction(y)
            for j in range(self.W_second.shape[0]):
                self.W_second[j] -= lmbd * delta[j] * out

            for j in range(self.W.shape[0]):
                sigma = sum(self.W_second[:, j] * delta)
                delta2 = sigma * activationFunctionDerivative(out[j])
                self.W[j] -= lmbd * delta2 * np.array(x)

    def startLearning(self):
        epoch = []
        true_output = []
        directories = ['greyscale/' + i + '/' for i in alphabet]
        N = 0
        for dir in directories:
            files = [f for f in listdir(dir) if isfile(join(dir, f))]
            N += len(files)
            i = 0
            for file in files:
                if i % 50 == 0:
                    print(f'{i}/{len(files)}')
                image = Image.open(f'{dir}{file}')
                image_array = np.asarray(image).tolist()

                

                res = []
                for x in image_array:
                    res.extend(x if isinstance(x, list) else [x])
                res1 = []
                for x in res:
                    res1.append(0 if x == [0, 0, 0, 0] else 1)


                epoch.append(res1)
                true_output.append(np.zeros((26, ), dtype=float))
                true_output[-1][ord(dir[len(dir) - 2]) - 97] = 1.
                i += 1

        print(len(epoch))
        self.Learn(epoch, true_output)
        
        print('Обучение завершилось')

        example = Image.open('test/a.png')
        example = np.asarray(example).tolist()
        res = []
        for x in example:
            res.extend(x if isinstance(x, list) else [x])
        example = []
        for x in res:
            example.append(0 if x == [0, 0, 0, 0] else 1)

        weights = {
            'W1': self.W.tolist(),
            'W2': self.W_second.tolist(),
        }

        with open('weights_100.json', 'w') as outfile:
            json.dump(weights, outfile, indent=4)
        print(len(epoch))


    def getResult(self, letter):
        self.W = json.load(open("weights_100.json", "r"))['W1']
        self.W_second = json.load(open("weights_100.json", "r"))['W2']
        example = Image.open(f'greyscale/{letter}.png')
        example = np.asarray(example).tolist()
        res = []
        for x in example:
            res.extend(x if isinstance(x, list) else [x])
        example = []
        for x in res:
            example.append(0 if x == [0, 0, 0, 0] else 1)

        print('Пример:')

        print(self.getCalculation(example)[0])
        answer = self.getCalculation(example)[0]
        max1 = 0
        index = 0
        for i in range(len(answer)):
            if answer[i] > max1:
                max1 = answer[i]
                index = i
        print(chr(index + 65))

        fig, ax = plt.subplots()

        x = np.array([chr(i) for i in range(65, 91)])
        y = np.array(answer)
        ax.bar(x, y)

        ax.set_facecolor('seashell')
        fig.set_facecolor('floralwhite')
        fig.set_figwidth(12)  # ширина Figure
        fig.set_figheight(6)  # высота Figure

        plt.show()
        
neu = Neuron()
# neu.startLearning()
print(neu.getResult('a12'))
