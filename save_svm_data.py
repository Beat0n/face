import scipy.io as scio
import os

dataset_path = "train(Task_2)"
train_data = scio.loadmat(os.path.join(dataset_path, "train_data.mat"))["train_data"]
train_label = scio.loadmat(os.path.join(dataset_path, "train_label.mat"))["train_label"]

with open('svm_train_data.txt', 'w') as w:
    for i in range(len(train_data)):
        w.write(str(train_label[i][0]) + " ")
        for j in range(len(train_data[i])):
            w.write(str(j+1) + ":" + '{:.4f}'.format(train_data[i][j]) + " ")
        w.write("\n")

test_data = scio.loadmat("test_data.mat")["test_data"]
with open('svm_test_data.txt', 'w') as w:
    for i in range(len(test_data)):
        w.write('0' + " ")
        for j in range(len(test_data[i])):
            w.write(str(j+1) + ":" + '{:.4f}'.format(test_data[i][j]) + " ")
        w.write("\n")