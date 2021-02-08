import numpy as np

#shape  lixia


#（batch，32，32，3）  (batch,1) lixia --》 img
def show_batch_img(batch_size,tp="train"):
    x_train=np.load("./data/x_train.npy")
    y_train=np.load()
    #[img1,img...]
    #[label,label2...]
    
    #[(img1,label1),(img2,label2),....]

    np.random.choice()

    


#（batch，32，32，3）  true (batch,1)  predict（batch,1) jiaqi
def show_batch_img(batch_size,,y_prd,tp="train"):
    x_train=np.load("./data/x_train.npy")
    y_train=np.load()

    #[img1,img...]
    #[label,label2...]
    
    #[(img1,label1,pred1),(img2,label2,pred2),....]

    np.random.choice()


#num-->str  jiaqi
def label2name(num):
    pass