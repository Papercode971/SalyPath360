import os  
import torch 
from model import Scanpath_based_Attention_module
import pandas as pd 
import numpy as np 
import cv2
from tqdm import tqdm
from torchvision import utils 

def get_name(path):
    return os.path.basename(path).split('.')[0]

def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 320))
    img = img.astype(np.float32)
    img = torch.FloatTensor(img)
    img = img.permute(2,0,1)  
    img =  img.unsqueeze(0)
    return img.cuda()

def load_init_model(weight_path,):
    model  = Scanpath_based_Attention_module()
    weight_= torch.load(weight_path)#['state_dict']
    model.load_state_dict(weight_,strict=True)
    model.cuda()
    return model


def save_sp_img(rootpath,sp,name,img,sal):
    
    sal = sal.squeeze().detach().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min())
    sal *= 255 

    print(sal.shape)
    heatmap_img = cv2.applyColorMap(sal.astype(np.uint8), cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.5, cv2.resize(img.astype(np.uint8),(640,320)), 0.5, 0)
    cv2.imwrite(os.path.join(rootpath,name+'_sal.jpg'),fin)
    
    
    os.makedirs(rootpath,exist_ok=True)
    sp = sp.detach().cpu().numpy().squeeze()
    sp = pd.DataFrame(sp,columns=['X', 'Y']) #to normalize use size 40,20 
    sp['X']  /= 40
    sp['Y']  /= 20 
    sp.to_csv(os.path.join(rootpath,name+'.csv'),index=False)

    h,w,_ = img.shape
    sp['X']  *= w
    sp['Y']  *= h 

    print(type(img))
    img_sp = cv2.polylines(img, np.int32([sp.values]),isClosed=False, color=(0,0,255) ,thickness = 1)

    print(type(img))
    print(img.shape)
    a = cv2.imwrite(os.path.join(rootpath,name+'.jpg'),img_sp)
    print(a)
    

# PATHS
imgs_paths = sorted(list(map(lambda x: os.path.join('/content/drive/MyDrive/full_aux_model/test_img/',x),os.listdir('/content/drive/MyDrive/full_aux_model/test_img/'))))
scanpath_paths = sorted(list(map(lambda x: os.path.join('/content/drive/MyDrive/full_aux_model/test/H/Scanpaths/',x),os.listdir('/content/drive/MyDrive/full_aux_model/test/H/Scanpaths/'))))
weight_path = '/content/drive/MyDrive/init_for_joint.pt' #'weight_mse_val_1/model_epoch_82.pt' 
save_path = '/content/drive/MyDrive/full_aux_model/test_results/'

model = load_init_model(weight_path)

for path in tqdm(imgs_paths):
    print(os.path.exists(path))
    print(path)
    img = load_img(path)
    sp, sal ,_ = model(img)
    name = get_name(path)
    save_sp_img(save_path,sp,name,img.squeeze().permute(1,2,0).cpu().numpy(),sal)
