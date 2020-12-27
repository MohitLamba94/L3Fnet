opt = {'LowLight_dir':'path/to/L3F-100/jpeg/folder'}
opt.update({'GT_dir':'path/to/L3F-1/jpeg/folder'})

opt['indv_blocks'] = 6 # number of blocks in stage II
opt['hidd_blocks'] = 4 # number of blocks in stage I

# We restore the central 8x8 views of the captured 15x15 LF
opt.update({'views_beg':5})
opt.update({'views_end':12})

opt['GPU'] = True

import random
import numpy as np
from PIL import Image
import imageio
import torchvision.transforms as transforms

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torchvision.transforms.functional as Ft

import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if opt['GPU']:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

class get_data(Dataset):
    """Loads the Data."""

    def __init__(self, opt, transforms=None):        
        
        self.transforms = transforms
        self.view_beg = opt['views_beg']
        self.view_end = opt['views_end']
        self.ip_dir = opt['LowLight_dir']
        self.op_dir = opt['GT_dir']
        
        self.files_op = sorted(os.walk(os.path.join(self.op_dir)))
        self.files_op = sorted(self.files_op[0][2])
        self.files_ip = sorted(os.walk(os.path.join(self.ip_dir)))
        self.files_ip = sorted(self.files_ip[0][2])        
        
    def __len__(self):
        return len(self.files_op)

    def __getitem__(self, idx):
        
        ## load images
        self.img_gt = Image.open(os.path.join(self.op_dir,self.files_op[idx])).convert('RGB')
        self.img = Image.open(os.path.join(self.ip_dir,self.files_ip[idx])).convert('RGB')
        
        ## patch generation
        width, height = self.img.size
        
        w = width//15
        h = height//15
        
        #we ignore the border by not considering `ignore' numbers of rows and columns in the beginning and end
        ignore = 0
        th = (h//2)*2 - ignore
        tw = (w//2)*2 - ignore       
        i = ignore
        j = ignore
        
        n_views = self.view_end-self.view_beg+1
        
        # We do not perform any data augmentation
        if random.randint(0, 100)>50:
            flip_flag = False
        else:
            flip_flag = False
            
        if random.randint(0, 100)<20:
            v_flag = False
        else:
            v_flag = False
            
        if random.randint(0, 100)>50:
            color_jitter_flag = False
            jitter_order = np.random.permutation(3)
        else:
            color_jitter_flag = False
        

        for ii in range(self.view_beg,self.view_end+1):
            for jj in range(self.view_beg,self.view_end+1):
                
                img_ip_small_c = Ft.crop(self.img,(i+(h*(jj-1))),(j+(w*(ii-1))),th,tw)
                img_ip_small_n = Ft.crop(self.img,(i+(h*(jj))),(j+(w*(ii-1))),th,tw)
                img_ip_small_e = Ft.crop(self.img,(i+(h*(jj-2))),(j+(w*(ii-1))),th,tw)
                img_ip_small_s = Ft.crop(self.img,(i+(h*(jj-1))),(j+(w*(ii))),th,tw)
                img_ip_small_w = Ft.crop(self.img,(i+(h*(jj-1))),(j+(w*(ii-2))),th,tw)
                    
                img_op_small = Ft.crop(self.img_gt,(i+(h*(jj-1))),(j+(w*(ii-1))),th,tw)
                    
                if flip_flag:
                    img_ip_small_c = Ft.hflip(img_ip_small_c)
                    img_ip_small_n = Ft.hflip(img_ip_small_n)
                    img_ip_small_e = Ft.hflip(img_ip_small_e)
                    img_ip_small_s = Ft.hflip(img_ip_small_s)
                    img_ip_small_w = Ft.hflip(img_ip_small_w)
                    
                    img_op_small = Ft.hflip(img_op_small)
                    
                if v_flag:
                    img_ip_small_c = Ft.vflip(img_ip_small_c)
                    img_ip_small_n = Ft.vflip(img_ip_small_n)
                    img_ip_small_e = Ft.vflip(img_ip_small_e)
                    img_ip_small_s = Ft.vflip(img_ip_small_s)
                    img_ip_small_w = Ft.vflip(img_ip_small_w)
                    
                    img_op_small = Ft.vflip(img_op_small)
                
                img_ip_small_c = self.transforms(img_ip_small_c) 
                img_ip_small_w = self.transforms(img_ip_small_w)
                img_ip_small_s = self.transforms(img_ip_small_s)
                img_ip_small_e = self.transforms(img_ip_small_e)
                img_ip_small_n = self.transforms(img_ip_small_n)
                
                img_op_small = self.transforms(img_op_small) 

                if color_jitter_flag:
                    img_ip_small_c = img_ip_small_c[jitter_order,...]
                    img_ip_small_w = img_ip_small_w[jitter_order,...]
                    img_ip_small_s = img_ip_small_s[jitter_order,...]
                    img_ip_small_e = img_ip_small_e[jitter_order,...]
                    img_ip_small_n = img_ip_small_n[jitter_order,...]

                    img_op_small = img_op_small[jitter_order,...]

                img_ip_small_concat = torch.cat([img_ip_small_c, img_ip_small_e, img_ip_small_n, img_ip_small_s, img_ip_small_w], dim = 0)
                img_ip_small_concat = torch.unsqueeze(img_ip_small_concat,0) 
                
                img_op_small = torch.unsqueeze(img_op_small,0)

                if ii==self.view_beg and jj==self.view_beg:
                    imgs_ip_indv = img_ip_small_concat
                    
                    imgs_ip_hid = img_ip_small_c 
                    
                    imgs_op = img_op_small 
                    
                else:
                    imgs_ip_indv = torch.cat((imgs_ip_indv,img_ip_small_concat),0)
                    
                    imgs_ip_hid = torch.cat((imgs_ip_hid,img_ip_small_c),0) 
                    
                    imgs_op = torch.cat((imgs_op,img_op_small),0) 
        #####
        permute = []
        
        imgs_ip_hid = torch.unsqueeze(imgs_ip_hid,0) 
        
        # LFs are very huge and may take lot RAM. Thus to save computation for your system only 2 SAIs are restored. Remove the below `choice' variable for restoring all 64 SAIs
        choice = np.asarray([12,35])

        return {'imgs_op':imgs_op[choice,...], 'imgs_ip_indv':imgs_ip_indv[choice,...], 'imgs_ip_hid':imgs_ip_hid}

trns = transforms.Compose([transforms.ToTensor()])
obj_train = get_data(opt, transforms=trns)
dataloader_train = DataLoader(obj_train, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)


class BasicBlock(nn.Module):
    
    def __init__(self):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity

        return out
    
model_ft = BasicBlock()
model_ftt = BasicBlock()

hidd_res_block_list = []
        
res_blocks_list = []

for i in range(opt['indv_blocks']):
    res_blocks_list.append(BasicBlock())
    
    if i<opt['hidd_blocks']:
        hidd_res_block_list.append(BasicBlock())
    
res_blocks = nn.Sequential(*res_blocks_list)
hidd_res_blocks = nn.Sequential(*hidd_res_block_list)

class Unet(nn.Module):
    def __init__(self, res_blocks, hidd_res_blocks):
        super(Unet, self).__init__()
        
        self.conv_pre_process_ind = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=7, stride=1, padding=3, bias=True),
            #nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.ReLU(inplace=True)
        ) 
        
        self.res_blocks = res_blocks
        self.hidd_res_blocks = hidd_res_blocks
        
        self.conv_pre_process_hidd = nn.Sequential(
            nn.Conv2d(in_channels=3*8*8, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            #nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.ReLU(inplace=True)
        ) 
        
        self.conv_post_process_hidd = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ) 
         
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,padding=0, output_padding=0),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x, h):
        
        identity = x[:,:3,:,:].clone()
        h = self.conv_post_process_hidd(self.hidd_res_blocks(self.conv_pre_process_hidd(h)))
        #print(h.size())
        h = torch.cat([h]*x.size()[0])
        
        x = self.conv_pre_process_ind(x)
        x = self.res_blocks(torch.cat([x,h], dim=1))
        x = self.deconv(x) + identity
        
        return x
    
obj = Unet(res_blocks,hidd_res_blocks)




class common_functions():
    
    def __init__(self, opt, res_blocks,hidd_res_blocks):
        
        self.opt = opt
        self.train_loss = 100
        self.count = 0
                        
        if opt['GPU']:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        model = Unet(res_blocks,hidd_res_blocks)
        
        self.model = model.to(self.device)
        print('Is model on GPU : ',next(self.model.parameters()).is_cuda)  
        
        
        checkpoint = torch.load('weights')
        self.model.load_state_dict(checkpoint['model'])
  
    
    def optimize_parameters(self, imgs_ip, imgs_hd, imgs_op):
        
        imgs_ip = imgs_ip.to(self.device)
        imgs_hd = imgs_hd.to(self.device)
        imgs_op = imgs_op.to(self.device)
        
        self.model.train()
        
        with torch.no_grad():    
            pred_output = self.model(imgs_ip, imgs_hd)
        
        print('Processed ',self.count+1,' number of images')
        self.count +=1
            
        plot_out_pred = torchvision.utils.make_grid(pred_output,nrow=8, padding=0, normalize=False)
        x = np.transpose(plot_out_pred.detach().cpu().numpy(),(1,2,0))
        minmaxnormalize = x#(x- np.min(x)) / (np.max(x) - np.min(x))
        plot_out_pred = (np.clip(minmaxnormalize,0,1)*255).astype(np.uint8) 
        
        plot_out_GT = torchvision.utils.make_grid(imgs_op,nrow=8, padding=0, normalize=False)
        x = np.transpose(plot_out_GT.detach().cpu().numpy(),(1,2,0))
        minmaxnormalize = x#(x- np.min(x)) / (np.max(x) - np.min(x))
        plot_out_GT = (np.clip(minmaxnormalize,0,1)*255).astype(np.uint8)
        
        
        # Save images
        imageio.imwrite('{}_IMG_PRED.png'.format(self.count),plot_out_pred)
        imageio.imwrite('{}_IMG_GT.png'.format(self.count),plot_out_GT)
            
full_model = common_functions(opt, res_blocks,hidd_res_blocks)

for iteration, img in enumerate(dataloader_train):
    full_model.optimize_parameters(img['imgs_ip_indv'][0],img['imgs_ip_hid'][0],img['imgs_op'][0])









