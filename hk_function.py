import numpy as np
import random 

def get_label(label): # 将label从长度10的one hot向量转换为0~9的数字
    return np.argmax(label)


def cross_validation(images, labels, k):
    X_train_images = []  #大小为k-1
    y_train_labels = []
    X_test_images=[]    #大小为1
    y_test_labels = []
    
    
    zero_images =[]
    zero_labels =[]
    one_images =[]
    one_labels =[]
    two_images =[]
    two_labels =[]
    three_images =[]
    three_labels =[]
    four_images =[]
    four_labels =[]
    five_images =[]
    five_labels =[]
    six_images =[]
    six_labels =[]
    severn_images =[]
    severn_labels =[]
    eight_images =[]
    eight_labels =[]
    nine_images =[]
    nine_labels =[]

    # 获取标签并分离
    for i in range(len(images)):
        if get_label(labels[i]) == 0:
            zero_images.append(images[i])
            zero_labels.append(labels[i])
        if get_label(labels[i]) == 1:
            one_images.append(images[i])
            one_labels.append(labels[i])
        if get_label(labels[i]) == 2:
            two_images.append(images[i])
            two_labels.append(labels[i])
        if get_label(labels[i]) == 3:
            three_images.append(images[i])
            three_labels.append(labels[i])
        if get_label(labels[i]) == 4:
            four_images.append(images[i])
            four_labels.append(labels[i])
        if get_label(labels[i]) == 5:
            five_images.append(images[i])
            five_labels.append(labels[i])
        if get_label(labels[i]) == 6:
            six_images.append(images[i])
            six_labels.append(labels[i])
        if get_label(labels[i]) == 7:
            severn_images.append(images[i])
            severn_labels.append(labels[i])
        if get_label(labels[i]) == 8:
            eight_images.append(images[i])
            eight_labels.append(labels[i])
        if get_label(labels[i]) == 9:
            nine_images.append(images[i])
            nine_labels.append(labels[i])
        
    total_images=[zero_images,one_images,two_images,three_images,four_images,
            five_images,six_images,severn_images,eight_images,nine_images]

    total_labels=[zero_labels,one_labels,two_labels,three_labels,four_labels,
            five_labels,six_labels,severn_labels,eight_labels,nine_labels]


    for i in range(10):
        k_total_images=[]
        k_total_labels=[]  #大小为k

        for j in range(k):
            k_total_images.append(total_images[i][int(j*len(total_images[i])/k):int((j+1)*len(total_images[i])/k)])  #长度为k*10，里面的列表长度为len(total_images[i])/k
            k_total_labels.append(total_labels[i][int(j*len(total_images[i])/k):int((j+1)*len(total_images[i])/k)])
    
        idex = random.randrange(0,k,1)
        X_test_images+=k_total_images[idex]  #1
        y_test_labels+=k_total_labels[idex]
        del k_total_images[idex]  #k-1
        del k_total_labels[idex]
        X_train_images+=k_total_images #k-1
        y_train_labels+=k_total_labels
        
    f_X_train_images=[]
    f_y_train_labels=[]
        
    for i in range(10*(k-1)):
        for j in range(len( X_train_images[i])):
            f_X_train_images.append(X_train_images[i][j]) #大小为k-1
            f_y_train_labels.append(y_train_labels[i][j])

            
    f_X_train_images,f_y_train_labels = np.array(f_X_train_images),np.array(f_y_train_labels)
    X_test_images,y_test_labels = np.array(X_test_images),np.array(y_test_labels)
    return (f_X_train_images,f_y_train_labels,X_test_images,y_test_labels)


def hold_out(images, labels, train_percentage):
    X_train_images = []
    y_train_labels = []
    X_test_images=[]
    y_test_labels = []

    zero_images =[]
    zero_labels =[]
    one_images =[]
    one_labels =[]
    two_images =[]
    two_labels =[]
    three_images =[]
    three_labels =[]
    four_images =[]
    four_labels =[]
    five_images =[]
    five_labels =[]
    six_images =[]
    six_labels =[]
    severn_images =[]
    severn_labels =[]
    eight_images =[]
    eight_labels =[]
    nine_images =[]
    nine_labels =[]


    for i in range(len(images)):
        if get_label(labels[i]) == 0:
            zero_images.append(images[i])
            zero_labels.append(labels[i])
        if get_label(labels[i]) == 1:
            one_images.append(images[i])
            one_labels.append(labels[i])
        if get_label(labels[i]) == 2:
            two_images.append(images[i])
            two_labels.append(labels[i])
        if get_label(labels[i]) == 3:
            three_images.append(images[i])
            three_labels.append(labels[i])
        if get_label(labels[i]) == 4:
            four_images.append(images[i])
            four_labels.append(labels[i])
        if get_label(labels[i]) == 5:
            five_images.append(images[i])
            five_labels.append(labels[i])
        if get_label(labels[i]) == 6:
            six_images.append(images[i])
            six_labels.append(labels[i])
        if get_label(labels[i]) == 7:
            severn_images.append(images[i])
            severn_labels.append(labels[i])
        if get_label(labels[i]) == 8:
            eight_images.append(images[i])
            eight_labels.append(labels[i])
        if get_label(labels[i]) == 9:
            nine_images.append(images[i])
            nine_labels.append(labels[i])
        
    total_images=[zero_images,one_images,two_images,three_images,four_images,
            five_images,six_images,severn_images,eight_images,nine_images]

    total_labels=[zero_labels,one_labels,two_labels,three_labels,four_labels,
            five_labels,six_labels,severn_labels,eight_labels,nine_labels]

    # 分离操作 前半部分训练集 后半部分测试集
    for i in range(10):
        X_train_images.append(total_images[i][:int(len(total_images[i])*train_percentage)])
        y_train_labels.append(total_labels[i][:int(len(total_images[i])*train_percentage)])
        X_test_images.append(total_images[i][int(len(total_images[i])*train_percentage):])
        y_test_labels.append(total_labels[i][int(len(total_labels[i])*train_percentage):])
   
    # 多维列表嵌套解开为1维
    fin_X_train_images=[]
    fin_y_train_labels=[]
    fin_X_test_images=[]
    fin_y_test_labels=[]
    for m in range(10):
        for n in range(len(X_train_images[m])):
            fin_X_train_images.append(X_train_images[m][n])
            fin_y_train_labels.append(y_train_labels[m][n])
        
    for m in range(10):
        for n in range(len(X_test_images[m])):
            fin_X_test_images.append(X_test_images[m][n])
            fin_y_test_labels.append(y_test_labels[m][n])
    fin_X_train_images,fin_y_train_labels = np.array(fin_X_train_images),np.array(fin_y_train_labels)
    fin_X_test_images,fin_y_test_labels = np.array(fin_X_test_images),np.array(fin_y_test_labels)
    return (fin_X_train_images,fin_y_train_labels,fin_X_test_images,fin_y_test_labels)