def function(x):
    return x[0]**2*x[1]+x[1]**3*x[0]

def superFunction(x):
    return function(x)*x[0]

def getGradient(f, x, y):
    d = 0.00005
    dirX = (f(x+d, y)-f(x-d, y)) / (2*d)
    dirY = (f(x, y+d)-f(x, y-d)) / (2*d)
    return [dirX, dirY]

def gradient(f, args):
    d = 0.0005
    result = []
    for i in range(len(args)):
        new_args = args.copy()
        new_args[i] += d
        first_f = f(new_args)
        new_args[i] -= 2*d
        second_f = f(new_args)
        result.append((first_f-second_f)/(2*d))
    return result
    
print(gradient(superFunction, [5, -3]))
print("end")