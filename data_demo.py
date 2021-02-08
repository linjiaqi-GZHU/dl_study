
#shape  lixia


#（batch，32，32，3）  (batch,1) lixia --》 img



#（batch，32，32，3）  true (batch,1)  predict（batch,1) jiaqi



#num-->str  jiaqi


'''

#题目

import numpy as np

#定义shape函数
def shape(dataset)  
	print (dataset.shape)
	return dataset.shape



'''





'''

题目
#(batch,32,32,3)   (batch,1)  --->img    lixia

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


'''




import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from math import ceil


#功能：将标签的数字索引转为字符串
def label2name(num):
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    return  labels[num]


#功能：显示定量训练集的图片
def show_batch_img(batch_size,tp="train"):                     
	x_train = np.load("./data/x_train.npy")
	y_train = np.load("./data/y_train.npy")       


	#训练图片合集
	img_set = x_train
	#标签合集
	label = y_train
	

	#对应绑在一起
	#[img1,img2......]
	#[label1,label2......]
	#[(img1,label1)(img2,label2),......]
	train_zip = list(zip(img_set, label))
	
	
	#随机抽取train_zip中的训练集
	#training = np.random.choice(train_zip, size=batch_size, replace=True)           
	training = random.sample(train_zip, batch_size)
	
	
	#画出training中的子图
	fig, axes = plt.subplots(ceil(batch_size/5), 5, figsize=(12,12))            
	print(type(axes[2][1]))
	print(axes.shape)
	axes = axes.flatten()
	for i in range(batch_size):
		axes[i].imshow(training[i][0])
		axes[i].set_title(label2name(training[i][1][0]))
		axes[i].axis('off')
	plt.tight_layout()                                                            
	plt.show()



	'''
	fig,axes = plt.subplots(ceil(batch_size/5),5,figsize=(12,12))
	ax = axes.ravel()
	for i in range(batch_size):
		ax[i].imshow(training[i][0])
		ax[i].set_title(label2name(training[i][1][0]))
	plt.tight_layout() 
	'''

		
#函数调用
show_batch_img(27,tp="train")






