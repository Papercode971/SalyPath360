import numpy as np
from scipy.io import savemat
from torch.utils import data
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from  model import Scanpath_based_Attention_module
from dataset import Dataset
import cv2 
import pandas as pd 

import logging
logging.basicConfig(filename='training_mse_val_2_all_sp2.log',level=logging.DEBUG)




def get_epoch(path):
    epoch = path.split('_')[-1]
    epoch = epoch.split('.')[0]
    return epoch




class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()

    #normalize saliency map
    def stand_normlize(self, x) :
       return (x - x.mean())/x.std()

    def forward(self, sal_map, fix):
        if sal_map.size() != fix.size():
           print("\n wronnnnng")
          
           sal_map = interpolate(sal_map, size= (fix.size()[1],fix.size()[0]))
           #print(sal_map.size())
           #print(fix.size())
        fix = fix > 0.5

        # Normalize saliency map to have zero mean and unit std
        sal_map = self.stand_normlize(sal_map)
        return sal_map[fix].mean()


import sys

from torch.nn.functional import interpolate 

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)




import torch.nn.functional as F

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.KLDLoss = KLDLoss()
        self.NSSLoss = NSSLoss()
        self.MSE = torch.nn.MSELoss()
        #self.state = state

    def FixMap(self,fixations):
        fixations = fixations[:,:2]
        fixmap = torch.zeros((20,40), dtype=torch.int)
        #print('int problem')
        for iFix in range(fixations.shape[0]):
          
          fixmap[ int(fixations[iFix, 1]), int(fixations[iFix, 0]) ] += 1
        fixmap[fixmap>0]=1
        return fixmap

    def forward(self,generated_scan_path,scan_path , generated_salmap,salmap):

        mse_lss = self.MSE(generated_scan_path,scan_path)
        kld_lss = self.KLDLoss(generated_salmap,salmap)
        reproduced_map = self.FixMap(generated_scan_path).cuda()
        #print("aaddfjkzhfl zrjfkmlq,klqjmdlelgkl")
       
        generated_salmap = F.interpolate(generated_salmap, size=(20,40))
        
        joint_loss = self.NSSLoss(generated_salmap.squeeze().cuda(),reproduced_map)
        #print(mse_lss,kld_lss,joint_loss)
        #return scan_path_loss/scan_path_loss + salmap_loss/salmap_loss + joint_loss
        return 0.01*mse_lss+0.8*kld_lss-0.19*joint_loss
          

def model_init_(weight_path,use_parallel = False,num_gpu = 2):
        model = Scanpath_based_Attention_module()
        weight_= torch.load(weight_path)
        #weight_= torch.jit.load(weight_path)
        model.load_state_dict(weight_,strict=False) 
        model.cuda()
        return model 


def dataloader_init_(): # erased root_path parameter
        training = Dataset() 
        print("Size of train set is {}".format(len(training)))
        train_loader = data.DataLoader(training, batch_size= 5,num_workers= 4,pin_memory= True)
        validation = Dataset() #Dataset(root='./test')
        print("Size of validation set is {}".format(len(validation)))
        validation_loader = data.DataLoader(validation, batch_size= 5,num_workers= 2,pin_memory= True)
        return train_loader,validation_loader

def calc_trend(epoch,val_list,TS):
  
        pass

def plot_scan_path(scan_path):
        
        pass 


def predict(model, img, epoch, path,writer=None):
    
    if not os.path.exists(path) :
        os.makedirs(path)

                                                # Transforms 0-255 numbers to 0 - 1.0.
    im = torch.cuda.FloatTensor(img)                    # transform image to Pytorch Tensor
    im = im.permute(2,0,1)
    out,sal,_ = model(im.unsqueeze(0))                            # output 
    out  =out[0].squeeze().detach().cpu().numpy()                              
    #print(out.shape)
    #print(out)
    new_path = path+"fix_" + str(epoch) + ".csv" 
    df = pd.DataFrame( np.squeeze(out),columns=['X','Y'])
    df.to_csv(new_path,index=False)             # saving output scanpath
    df['X'] = df['X'] / 40  * img.shape[1]
    df['Y'] = df['Y'] / 20 * img.shape[0]

    new_path = path+ 'img_' + str(epoch) + ".png"
    #print(np.int32(out))
    #print(np.int32(out).shape)
    img = cv2.polylines(img, np.int32([out]),isClosed=False, color=(0,0,255) ,thickness = 1)

    salmap =  sal.squeeze().detach().cpu().numpy()  
    if not writer == None :
        writer.add_image(path,np.int8(img),epoch) 
    cv2.imwrite(new_path,img)
    cv2.imwrite(new_path.replace('img','sal'),255*salmap/salmap.max())





model = model_init_(weight_path = '/content/drive/MyDrive/init_for_joint.pt')
train_loader,validation_loader = dataloader_init_()

train_loss = []
val_loss   = []



#optimizer = torch.optim.Adam([
#    {'params': model.decoder_hm.parameters(), 'lr': 1e-5,'weight_decay': 1e-6},{'params': model.aux.parameters(), 'lr': 1e-5,'weight_decay': 1e-6},{'params':model.decoder.parameters(), 'lr':1e-5}],lr=1e-6)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, amsgrad=True)
certirion = LOSS()

for epoch in range(100):
    model.set_freez(epoch)
    if (epoch+1)%4 == 0:
      optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
    # statrt trainging 
    epoch_loss = []
    epoch_loss_nss = []
    epoch_loss_kld = []
    epoch_loss_mse = []
    batch_index = 0
    train_loader = tqdm(train_loader)
    for batch in train_loader:
        batch_index += 1
        
        batch_loss1 = torch.Tensor([0.0])
        batch_loss1_list = []

        batch_loss  = torch.Tensor([0.0])

        batch_loss2 = torch.Tensor([0.0])
        batch_loss2_list = []
        
        batch_loss3 = torch.Tensor([0.0])
        batch_loss3_list = []

        pnan_batch,sp_batch,img_batch ,sm_batch= batch
        len_batch = len(img_batch)
        #print(len_batch)
        for idx in range(len_batch):
              #print(len(batch[idx]))
              pnan,sp,sm,img = pnan_batch[idx],sp_batch[idx],sm_batch[idx],img_batch[idx]
              sp = sp.squeeze().cuda()
              sm = sm.squeeze().cuda()
              img = img.squeeze().cuda()
              #print(img.shape)
              out_path ,out_saliency, _ = model(img.unsqueeze(0))
              #img_loss = certirion(out_path.squeeze(), sp.squeeze())
              #batch_loss = batch_loss.item() + img_loss
              #print('\n about  out values :',np.isnan(np.array(out_path.detach().cpu())).any())
              #print(pnan)
              lss  = certirion(out_path.squeeze(), sp,out_saliency,sm)
              batch_loss = batch_loss.item() + lss
              #lss1,lss2,lss3 = certirion(out_path.squeeze(), sp,out_saliency,sm)
              #print('\n',lss3.detach().cpu())
              #batch_loss1 = batch_loss1.item() + lss1
              #batch_loss1_list.append(lss1.detach().cpu().numpy())

              #batch_loss2 = batch_loss2.item() + lss2
              #batch_loss2_list.append(lss2.detach().cpu().numpy())

              #batch_loss3 = batch_loss3.item() + lss3
              #batch_loss3_list.append(lss3.detach().cpu().numpy())
        
        #batch_loss1 = (batch_loss1- np.array(batch_loss1_list).mean())/np.array(batch_loss1_list).std()
        #batch_loss2 = (batch_loss2- np.array(batch_loss2_list).mean())/np.array(batch_loss2_list).std()
        #batch_loss3 = (batch_loss3- np.array(batch_loss3_list).mean())/np.array(batch_loss3_list).std()

        #tloss = batch_loss1+batch_loss2-0.2*batch_loss3
        tloss = batch_loss
        #if epoch < 5 :
        #    tloss = 0.1*batch_loss1-0.8*batch_loss2+0.1*batch_loss3
        #else :
        #    tloss = 0.3*batch_loss1-0.6*batch_loss2-0.1*batch_loss3
        train_loader.set_description(f'batch loss : {tloss.data : 4.9f} ')
        tloss.backward()
        optimizer.step()
        epoch_loss.append(tloss.data)


        # بدل lenght تفكر وش قالك امين 
        #epoch_loss_nss.append(sum(batch_loss3_list)/5)
        #epoch_loss_kld.append(sum(batch_loss2_list)/5)
        #epoch_loss_mse.append(sum(batch_loss1_list)/5)
        
        
    # epoch loss 
    #print("********************************************************************")

    print("Final training epoch loss  Epoch {}/100 == {}".format(epoch,sum(epoch_loss)/batch_index))
    #print("Final training epoch loss nss  Epoch {}/100 == {}".format(epoch,sum(epoch_loss_nss)/batch_index))
    #print("Final training epoch loss kld  Epoch {}/100 == {}".format(epoch,sum(epoch_loss_kld)/batch_index))
    #print("Final training epoch loss mse  Epoch {}/100 == {}".format(epoch,sum(epoch_loss_mse)/batch_index))



    logging.info("Final training epoch loss  Epoch {}/100 == {}".format(epoch,sum(epoch_loss)/batch_index))
    #print("********************************************************************")
    train_loss.append(sum(epoch_loss)/batch_index)      

    img = cv2.imread('/content/drive/MyDrive/full_aux_model/P94_5376x2688.jpg')
    img = cv2.resize(img, (640, 320))

    os.makedirs("./viz_mse_val_2_all_sp2",exist_ok=True)
    predict(model, img, epoch, './viz_mse_val_2_all_sp2/',writer=None)
    print('epoch test saved' )
    """
    # statrt validation 
    epoch_loss = []

    batch_index = 0
    for batch in tqdm(validation_loader):
        batch_index += 1
        
        batch_loss1 = torch.Tensor([0.0])
        batch_loss1_list = []
        
        batch_loss2 = torch.Tensor([0.0])
        batch_loss2_list = []
        
        batch_loss3 = torch.Tensor([0.0])
        batch_loss3_list = []

        pnan_batch,sp_batch,img_batch ,sm_batch= batch
        len_batch = len(img_batch)
        #print(len_batch)
        for idx in range(len_batch):


              sp,sm,img = sp_batch[idx],sm_batch[idx],img_batch[idx]
              sp = sp.squeeze().cuda()
              sm = sm.squeeze().cuda()
              img = img.squeeze().cuda()


              with torch.no_grad():
                  out_path ,out_neck ,_= model(img.unsqueeze(0))

              lss1,lss2,lss3 = certirion(out_path.squeeze(), sp,out_saliency,sm)

              batch_loss1 = batch_loss1.item() + lss1
              batch_loss1_list.append(lss1.detach().cpu().numpy())

              batch_loss2 = batch_loss2.item() + lss2
              batch_loss2_list.append(lss2.detach().cpu().numpy())

              batch_loss3 = batch_loss3.item() + lss3
              batch_loss3_list.append(lss3.detach().cpu().numpy())
              
        batch_loss1 = (batch_loss1- np.array(batch_loss1_list).mean())/np.array(batch_loss1_list).std()
        batch_loss2 = (batch_loss2- np.array(batch_loss2_list).mean())/np.array(batch_loss2_list).std()
        batch_loss3 = (batch_loss3- np.array(batch_loss3_list).mean())/np.array(batch_loss3_list).std()

        tloss = batch_loss1+batch_loss2-batch_loss3
        # print("Validation loss for : Epoch {}/100, Image {} == {}".format(epoch,batch_index,batch_loss.data/(len_batch)))    
        epoch_loss.append(tloss.data)
    
    # print("--------------------------------------------------------------------")   
    # print("Final validation epoch loss  Epoch {}/100 == {}".format(epoch,sum(epoch_loss)/batch_index))
    logging.info("Final validation epoch loss  Epoch {}/100 == {}".format(epoch,sum(epoch_loss)/batch_index))    
    # print("--------------------------------------------------------------------") 
    # val_loss.append(sum(epoch_loss)/batch_index)
    """
    # save epoch loss 
    os.makedirs("./weight_mse_val_2_all_sp2",exist_ok=True)
    torch.save({'epoch': epoch + 1,'state_dict': model.state_dict()}, "./weight_mse_val_2_all_sp2/model_epoch_{}.pt".format(epoch))
    ##plot_saved = {"train_loss": train_loss, "val_loss": val_loss}
    ##savemat("./plot_dict", plot_saved)
    # Viszualization 

print('*********************************************')
#print(f'Global mean Train loss: {sum(train_loss) / len(train_loss)},  Global mean Val Loss : {sum(val_loss) / len(val_loss)}')
print(f'Global range Train loss: {max(train_loss) - min(train_loss)}')#',  Global range Val Loss : {max(val_loss) - min(val_loss)}')
print(f'Global change Train loss: {train_loss[0] - train_loss[-1]}')# ',  Global change Val Loss : {val_loss[0] - val_loss[-1]}')

#print(f'Global change Train loss: {np.std(train_loss)},  Global change Val Loss : {np.std(val_loss)}')

print('*********************************************')
