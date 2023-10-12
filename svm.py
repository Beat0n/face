from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import svm_predict, svm_train, svm_save_model

def store_txt(label, result_txt):
    with open(result_txt, 'w') as w:
        for index, result in enumerate(label, 1):
            w.write(str(index) + " " + str(result) + "\n")

train_label, train_pixel = svm_read_problem("svm_train_data.txt")
test_label, test_pixel = svm_read_problem("svm_test_data.txt")
model = svm_train(train_label, train_pixel, '-s 1 -t 2 -c 1 -g 0.8')
svm_save_model("svm.pth", model)
p_label, p_acc, p_val = svm_predict(train_label, train_pixel, model)
print(p_acc)
p_label, _, p_val = svm_predict(test_label, test_pixel, model)

store_txt(p_label, 'svm_result')
