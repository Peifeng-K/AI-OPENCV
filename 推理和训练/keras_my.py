'''

Python 3.7 (tensorflow) D:\Anaconda\install\envs\tensorflow\python.exe

keras
1.获取训练数据，测试数据
2.搭建模型
3.对数据进行预处理
4.用搭建好的模型训练
5.测试
'''

'''
1.获取训练数据，测试数据
'''
from tensorflow.keras.datasets import mnist #tensorflow自带的数据集
import matplotlib.pylab as plt
#加载数据集
(train_img,train_label),(test_img,test_label)=mnist.load_data()
print('train_img.shap = ',train_img.shape)
print('train_label = ',train_label)
print('test_img.shap = ',test_img.shape)
print('test_label = ',test_label)

digit = test_img[0]
print(digit.shape)
#print(digit)

plt.imshow(digit,cmap=plt.cm.binary)
plt.show()


'''
2.搭建模型
'''
from tensorflow.keras import models #
from tensorflow.keras import layers #数据处理层

#搭建顺序模型
network = models.Sequential()
#添加隐藏层
network.add(layers.Dense(10,activation = 'softmax',input_shape=(28*28,)))
#network.add(layers.Dense(10,activation = 'softmax'))

#model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
network.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics=['accuracy'])

'''
3.对数据进行预处理
'''
train_img = train_img.reshape((60000,28*28))
train_img = train_img.astype('float32')/255 #归一化，因为每个像素值取值范围为0~255

test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype('float32')/255

#print(train_img[0])

#one-hot
from tensorflow.keras.utils import to_categorical
print("before change:",test_label[0])
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
print("after change: ",test_label[0])



'''
4.用搭建好的模型训练
'''
network.fit(train_img,train_label,epochs = 5,batch_size=128)

#评估模型,不输出预测结果
#输入数据和标签,输出损失和精确度;
test_loss,test_acc=network.evaluate(test_img,test_label,verbose=1)
print('test_loss',test_loss)
print('test_acc:',test_acc)

'''
5.测试
'''

(train_img,train_label),(test_img,test_label)=mnist.load_data()
digit = test_img[715]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
test_img = test_img.reshape((10000,28*28))

#输入测试数据,输出预测结果
res = network.predict(test_img)

print(res[715])
print(res[715].shape[0])

for i in range(res[715].shape[0]):
    if (res[715][i]==1):
        print("the number for picture is: " ,i)
        break


