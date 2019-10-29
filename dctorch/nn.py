import sys,torch,torchvision,logging
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
#logging.basicConfig(level=logging.DEBUG)
logging.info("torch: {0} ,  torchvision:{1}".format(torch.__version__,torchvision.__version__))
class Learner(object):
    def __init__(self,train_dataset,test_dataset,model):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        #torch的Dataset
        self.__train_dataset = None
        self.__test_dataset = None
        #优化器
        self.__optimizer =None
        self.__optim_param={}
        #损失函数
        self.__criterion=None
        self.__crit_param={}
        #训练相关
        self.__train_param={}
        self.__train_dataloader=None
        #测试相关
        self.__test_param={}
        self.__test_dataloader=None
        # 模型
        self.__model=None
        #
        self.__device=torch.device("cpu")
        self.__trained=False
        if (isinstance(train_dataset,Dataset)):
            self.__train_dataset=train_dataset
        else:
            raise(TypeError("dataset必须为pythrch Dataset类型"))
        if (isinstance(test_dataset,Dataset)):
            self.__test_dataset=test_dataset
        else:
            raise(TypeError("dataset必须为pythrch Dataset类型"))
        if (isinstance(model,nn.Module)):
            self.__model=model
        else:
            raise(TypeError("model必须为pytorch Module类型"))
    def setCriterion(self,crit,**param):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        #设置损失函数
        #https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
        if issubclass(crit,nn.modules.loss._Loss):
            self.__criterion=crit
            self.__crit_param=param
        else:
            raise(TypeError("损失函数必须是 _Loss 子类"))
            #crit(**param)
    def setOptimizer(self,opt,**param):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        #设置优化器
        if issubclass(opt,torch.optim.Optimizer):
            self.__optimizer=opt
            self.__optim_param=param
        else:
            raise(TypeError("优化器必须是 Optimizer 子类"))

    def setTrainParam(self,key,value):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
         # 设置训练的参数
        self.__train_param[key]=value
    def setTestParam(self,key,value):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
         # 设置训练的参数
        self.__test_param[key]=value     

    def checkDevice(self,device="auto"):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        #检查cpu还是gpu
        if device=="auto":
            if torch.cuda.is_available():
                self.__device=torch.device("cuda")
        elif device=="cuda":
                self.__device=torch.device("cuda")
        logging.info("Use Device:{}".format(self.__device))

    def __checkTrainParam(self):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        # 检查参数 lr,bs
        check=set(['lr','bs'])
        print(self.__train_param.keys())
        if not (check <= self.__train_param.keys()):
            raise(ValueError("必须要设置lr和bs"))
    def showSummary(self):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        #打印汇总
        print("Device:{}".format(self.__device))
        print("Optimizer:{},Param:{}".format(self.__optimizer,self.__optim_param)) 
        print("Criterion:{},Param:{}".format(self.__criterion,self.__crit_param)) 
        print("TrainParam:{}".format(self.__train_param))   

        

    def train(self,fetch,epoches=10,device="auto",show_summary=False):
    
        logging.debug("----"+sys._getframe().f_code.co_name+"----")
        self.__losses=[]
        self.__trained=True
        #检查参数
        self.__checkTrainParam()
        #读取数据,默认不打乱
        shuffle= self.__train_param["shuffle"] if "shuffle" in self.__train_param.keys() else False
        self.__train_dataloader=DataLoader(dataset = self.__train_dataset, 
                                            batch_size = self.__train_param['bs'], 
                                            shuffle = shuffle)  # 将数据打乱
        #判断gpu
        self.checkDevice(device=device)

        # 显示汇总信息
        if (show_summary):
            self.showSummary()

        # 模型
        self.__model=self.__model.to(self.__device)
        self.__model.train()
        # 优化器
        train_opti=self.__optimizer(self.__model.parameters(),**self.__optim_param)
        # 损失函数
        train_crit=self.__criterion(**self.__crit_param).to(self.__device)
        for epoch in range(epoches):
            #循环
            for i,data in enumerate(self.__train_dataloader):
                logging.info("epoch:{},batch:{}".format(epoch+1,i))
                #批次
                x,y=fetch(data)
                train_opti.zero_grad()
                y_hat= self.__model(x.to(self.__device))
                loss = train_crit(y_hat, y.to(self.__device))
                loss.backward()
                train_opti.step()
                self.__losses.append(loss.cpu().data.item())
            print("epoch:{},avg loss:{}".format(epoch+1,np.mean(self.__losses)))
        print("训练完成")
    def test(self,fetch,metrics):
        logging.debug("----"+sys._getframe().f_code.co_name+"----")

        if  self.__trained==False :
            print("请先运行train")
            return
        bs= self.__test_param["bs"] if "bs" in self.__test_param.keys() else self.__train_param['bs']
        self.__test_dataloader=DataLoader(dataset = self.__test_dataset, 
                                            batch_size = bs, 
                                            shuffle = False) 
        self.__model=self.__model.to(self.__device)
        self.__model.eval()
        correct =total = 0
        for data in self.__test_dataloader:
            inputs,labels=fetch(data)
            outputs = self.__model(inputs.to(self.__device)).cpu()
            predicted = metrics(outputs)
            total+=labels.size(0)
            correct+=(predicted == labels).sum()
        print('准确率: %6.f %%' % (100 * correct / total))
    def saveModel(self,save_path):
        torch.save(self.__model.state_dict(), save_path)
    def loadModel(self,load_path):
        self.__model.load_state_dict(torch.load(load_path))
    def showLossPlt(self):
        plt.xlabel('batch #')
        plt.ylabel('Loss')
        plt.plot(self.__losses)
        plt.show()
 
        

        