
import torch
import os

from .func import start, finish, time2hms


# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print(f"\n=== Using {device}({__name__}). ===\n")


class MyLoggerTrain():
    def __init__(self, model, model_save, model_save_path, n_epoch, batch, epoch_now) -> None:
        #print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(f"It is scheduled to make model learn with {batch} batches, {n_epoch} epochs.")
        #print(f"Now, {epoch_now} epoch is already learned.")
        #print(f"And {device} device is available.")
        #print(f"Start learning!")
        #print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.model_save = model_save
        self.model_save_path = model_save_path
        self.n_epoch = n_epoch
        self.batch = batch
        self.model = model
    
    def s_train(self):
        self.start_train = start()
    def f_train(self):
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        train_time = finish(self.start_train) # trainの時間計測終了
        ave = train_time/self.n_epoch
        print(f"Time taken to study {self.n_epoch} epoch : {time2hms(train_time)}")
        print(f"Average time per epoch : {time2hms(ave)}")
    
    def s_epoch(self, epoch):
        print(f"\nEpoch : {epoch}")
        print("========================================================================\n")
        self.start_epoch = start() # epochの時間計測開始
        self.epoch = epoch
    def f_epoch(self):
        epoch_time = finish(self.start_epoch) # epochの時間計測終了
        print(f"Epoch {self.epoch} time : {time2hms(epoch_time)}\n")
        if self.epoch % 5 == 0 and self.model_save:
            self.model.save(self.model_save_path)
        print(f"\n========================================================================")
    

class MyLoggerModel:
    def __init__(self) -> None:
        pass
    def s_dataload(self):
        self.start_loading = start()  # loadingの時間計測開始
        print(f"\tLoading dataset...")
    def f_dataload(self):
        loading_time = finish(self.start_loading) # loadingの時間計測終了
        print(f"\tdataset was loaded!")
        print(f"\t* Loading time is {loading_time} sec. *")
    
    def s_forward(self):
        self.start_forward = start() # forwardの時間計測開始
        print(f"\tForwarding...")
    def f_forward(self):
        forward_time = finish(self.start_forward) # forwardの時間計測終了
        print(f"\tForward propagation was finished!")
        print(f"\t= Forward propagation time is {forward_time} sec. =")
    
    def s_backward(self):
        self.start_backward = start() # backwardの時間計測開始
        print(f"\tBackwarding...")
    def f_backward(self):
        backward_time = finish(self.start_backward) # backwardの時間計測終了
        print(f"\tBackward propagation was finished!")
        print(f"\t= Backward propagation time is {backward_time} sec. =")