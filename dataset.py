import random
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils import data
from PIL import Image
from torchvision import utils

class Dataset(data.Dataset):

    def __init__(self, root = './dataset/Images' , train=True, transform=None):
        #data_dir = os.path.join(root, 'train' if train else 'eval')
        data_dir = root
        self.transform = transform
        self.train = train
        target_ScanPath = []
        files_sorted = sorted(os.listdir(data_dir+'/H/Scanpaths'), key = lambda x: (x.split(".")[0])[10:12])
        # files_sorted = files_sorted[:3] 
        frame_files_sorted = sorted(os.listdir(data_dir+'/stimuli'), key = lambda x: (x.split(".")[0])[1:3]) # complexity for the sake of complexity 
        # frame_files_sorted = frame_files_sorted[:3]
        i = 0
        print(type(target_ScanPath))
        for file in tqdm(files_sorted, desc='scanning dir'):
            if file.endswith('.txt'):
              
              scan_path_path =os.path.join(data_dir,'H/Scanpaths', file)
              scan_path = np.loadtxt(scan_path_path, delimiter=",", skiprows=1, usecols=(1,2))
              
              #for idx in range(int(scan_path.shape[0]/100)):
              for idx in range(1,int(2)):
                  # idx = int(idx)
                  user_path_indx = (idx*100,idx*100+100)
                  image_path = os.path.join(data_dir,'stimuli', frame_files_sorted[i])
                  a = (user_path_indx,scan_path_path,image_path)
                  target_ScanPath.append(a)
                  #print(target_ScanPath)
            i += 1
        #print(target_ScanPath)        
        random.shuffle(target_ScanPath)
        self.target_ScanPath = target_ScanPath
        #print(target_ScanPath)        
            

    def __getitem__(self, item):

        user_path_indx ,scan_path_path ,image_path =  self.target_ScanPath[item]
        #img = np.array(Image.open(image_path))
        
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640, 320))
        img = img.astype(np.float32)
        
        img = torch.FloatTensor(img)
        img = img.permute(2,0,1)


        #sal_map_path = image_path.replace("stimuli", "H/SalMaps")
        sal_map_path = scan_path_path.replace('Scanpaths/Hscanpath','SalMaps/Hsalmap')
        sal_map_path = sal_map_path.replace('txt','png')


        salmap = cv2.imread(sal_map_path,0)
        salmap = cv2.resize(salmap, (640, 320))
        salmap = salmap.astype(np.float32)

        salmap = (salmap-np.min(salmap))/(np.max(salmap)-np.min(salmap))
        salmap = torch.FloatTensor(salmap)


        scan_path = np.loadtxt(scan_path_path, delimiter=",", skiprows=1, usecols=(1,2))
        user_path = scan_path[user_path_indx[0]:user_path_indx[1],:] 
        user_path = user_path * [40, 20] 
        user__path = np.zeros_like(user_path)
        user__path[:,0]=user_path[:,1]
        user__path[:,1]=user_path[:,0]
        #fixmap = self.FixMap(scan_path)
        if self.transform:
            img = self.transform_image(img)
            #fixmap = self.transform_image(fixmap)
            user_path = self.transform_path(user_path)
            
        # Image
        # perefer to put dataset mean 
        
        
        # FixMap
        #fixmap = fixmap.astype(np.float32).reshape(1,20,40)
        #fixmap = torch.FloatTensor(fixmap)
        # User_Path 
        user_path = torch.FloatTensor(user__path)
        return (image_path,user_path,img,salmap)


    def FixMap(self,fixations):

        fixations = fixations[:,:2]
        fixations = fixations * [40, 20] - [1,1]; fixations = fixations.astype(int)
        fixmap = np.zeros((20,40), dtype=int)
        for iFix in range(fixations.shape[0]):
          fixmap[ fixations[iFix, 1], fixations[iFix, 0] ] += 1
        fixmap[fixmap>0]=1
        return fixmap
    
    def __len__(self):
        return len(self.target_ScanPath)


"""
from torchvision import  utils
training = Dataset()
#print("Size of train set is {}".format(len(training)))
train_loader = data.DataLoader(training, batch_size= 1,num_workers= 1,pin_memory= True)
print(type(train_loader),type(training))

user_path ,fixmap ,img = next(iter(train_loader))

print(type(user_path))
print(type(fixmap))
print(type(img))
pritn(user_path)
utils.save_image(img, "/content/img.png")
utils.save_image(fixmap, "/content/fix.png")
"""
