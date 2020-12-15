import numpy as np
train = []
valid = []
test = []
data = np.genfromtxt("15_game_reg.txt", dtype=None)


for i in data:
    if "train" in i[0].decode('UTF-8'):
        train.append(i[2])
    elif "valid" in i[0].decode('UTF-8'):
        valid.append(i[2])
    else:
        test.append(i[2])

print(np.mean(train))
print(np.std(train))

print(np.mean(valid))
print(np.std(valid))

print(np.mean(test))
print(np.std(test))