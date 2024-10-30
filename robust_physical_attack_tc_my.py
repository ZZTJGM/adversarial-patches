import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('./detection')
sys.path.append('./yolov2')
sys.path.append('./models')
sys.path.append('./yolov2/utils')
import PIL.Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

import kornia
from kornia.geometry.bbox import bbox_generator,transform_bbox
from kornia.losses import total_variation
from kornia.enhance.adjust import adjust_brightness
# from detection.models.faster_rcnn import fasterrcnn_resnet50_fpn
from random import choice
from scipy.stats import rv_discrete

from yolov2.models.yolov2_d19 import YOLOv2D19 as yolo_net
from yolov2.data import config as yolo_config
from yolo_rt.config.config_test import args_test
from yolo_rt.config import build_model_config
from yolo_rt.models.detectors import build_model
from yolo_rt.utils.misc import load_weight
import torchvision.ops.boxes as bbox_tc

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from kornia.color import rgb_to_lab,lab_to_rgb,rgb_to_ycbcr,ycbcr_to_rgb
import random  
import joblib
import socket  

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa
    
def generate_gt(boxes,tansform_mat=None,gt_size=416):
    if not tansform_mat is None:
        boxes=transform_bbox(tansform_mat,boxes)
    boxes=[torch.tensor(([gt[0,:,0].min(),gt[0,:,1].min(),gt[0,:,0].max(),gt[0,:,1].max()])).reshape(-1,4).cuda() for gt in boxes]
    return boxes

def showarray(mat, name):
    # return 0
    pic = transforms.ToPILImage()(mat)
    pic.save(name+'.png')

def integrate_box_label(boxes_all,labels_all,model_name):
    if 'FasterRCNN_COCO' in model_name:
        num_classes=91
    elif 'YOLOV2_COCO' in model_name:
        num_classes=80
    elif 'YOLOV8_TT100K' in model_name:
        num_classes=45
    result_all=[]
    for boxes,labels in zip(boxes_all,labels_all):
        result=[[] for i in range(num_classes)]
        for i,label in enumerate(labels):
            result[label].append(boxes[i])
    
        for i in range(len(result)):
            if len(result[i])>0:
                result[i]=torch.vstack(result[i]).detach().cpu().numpy()
            else:
                result[i]=np.empty((0,5),dtype=np.float32)
        result_all.append(result)
    return result_all
    
def get_pred_for_grad(model,imgs):
    if 'FasterRCNN_COCO' in model.name:
        detections,proposal_list=model(list(imgs['img']))

        for box,det in zip(proposal_list,detections):
            if len(box)!=len(det['class_logits']):
                print(1)
        return [p/800 for p in proposal_list],[proposal['class_logits'] for proposal in detections],[torch.hstack([dets['boxes'],dets['scores'].reshape(-1,1)]) for dets in detections],[dets['labels'] for dets in detections]
    elif 'YOLOV2_COCO' in model.name:
        scores=[]
        bboxes=[]
        bbox_scores=[]
        det_cls_inds=[]
        img_mean=imgs['img_metas']['mean'][None,:,None,None]
        img_std=imgs['img_metas']['std'][None,:,None,None]
        img_size=imgs['img'][0].shape[-1]
        scores, obj_scores, bboxes, det_bboxs, det_scores, det_cls_inds=model.inference_with_grad((imgs['img']-img_mean)/img_std)
        bbox_scores=[]
        for  det_bbox,det_score in zip(det_bboxs,det_scores):
            bbox_scores.append(torch.from_numpy(np.hstack([det_bbox.reshape(-1,4)*img_size,det_score.reshape(-1,1)])))
        return bboxes,scores,bbox_scores,det_cls_inds
    elif 'YOLOV8_TT100K' in model.name:
        scores=[]
        bboxes=[]
        bbox_scores=[]
        det_cls_inds=[]
        img_size=imgs['img'][0].shape[-1]
        imgs['img'][:,[0,2],:,:]=imgs['img'][:,[2,0],:,:]
        scores, bboxes, det_bboxs, det_scores, det_cls_inds = model.forward_with_grad(imgs['img'])
        bbox_scores=[]
        for  det_bbox,det_score in zip(det_bboxs,det_scores):
            bbox_scores.append(torch.from_numpy(np.hstack([det_bbox.reshape(-1,4),det_score.reshape(-1,1)])))
        return bboxes/416,scores,bbox_scores,det_cls_inds

    else:
        raise ValueError ('Wrong model')
    
def inference(model, images, gts,gt_label,conf_thresh,flag_loss,b_relight=False): #写一个带patch_transforms的inference
    bbox_obj_scores,cls_scores,det_boxes,det_labels=get_pred_for_grad(model,images)
    vs=splits(gts,bbox_obj_scores,cls_scores,gt_label,conf_thresh,flag_loss)
    boxes_labels=integrate_box_label(det_boxes,det_labels,model.name)    
    return vs, boxes_labels   

def rgb_to_other(imgs,mode='ycbcr'):
    assert(4==len(imgs.shape))
    assert(imgs.shape[-1]==imgs.shape[-2])
    if 'ycbcr'==mode:
        imgs_out=rgb_to_ycbcr(imgs)
    elif 'lab'==mode:
        imgs_out=rgb_to_lab(imgs)
    elif 'rgb'==mode:
        imgs_out=imgs
    else:
        raise ValueError('Wrong color space')
    return imgs_out

def other_to_rgb(imgs,mode='ycbcr'):
    assert(4==len(imgs.shape))
    assert(imgs.shape[-1]==imgs.shape[-2])
    if 'ycbcr'==mode:
        imgs_out=ycbcr_to_rgb(imgs)
    elif 'lab'==mode:
        imgs_out=lab_to_rgb(imgs)
    elif 'rgb'==mode:
        imgs_out=imgs
    else:
        raise ValueError('Wrong color space')
    return imgs_out

def relight(bg_imgs,sign_imgs,obj_masks,method='my'):

    if 'my'==method:
        '''
        根据均值调整L通道
        '''
        num_imgs=bg_imgs.shape[0]
        coeffs = torch.zeros(num_imgs, 2).cuda()

        bg_imgs_lab=rgb_to_other(bg_imgs,'lab')
        sign_imgs_lab=rgb_to_other(sign_imgs,'lab')

        for i in range(num_imgs):
            if obj_masks[i].sum()<=0:
                continue
            real_pixels=torch.masked_select(bg_imgs_lab[i,0], obj_masks[i,0].to(torch.bool))
            sign_pixels=torch.masked_select(sign_imgs_lab[i,0], obj_masks[i,0].to(torch.bool))
            mean_real = real_pixels.mean()
            mean_sign = sign_pixels.mean()
            coeffs[i] = torch.tensor([mean_real,mean_sign], dtype=torch.float32).reshape(-1,2)
        mean_sign=sign_imgs_lab[:,0,:,:].mean(axis=-1).mean(axis=-1).reshape(-1)
        relighted_sign_imgs_lab=torch.concat([sign_imgs_lab[:,0,:,:].unsqueeze(1)-coeffs[:,1][:,None,None,None]+coeffs[:,0][:,None,None,None],
                                            sign_imgs_lab[:,1:,:,:]],dim=1)
        relighted_sign_imgs=other_to_rgb(relighted_sign_imgs_lab,'lab')
    elif 'reap'==method:
        '''
        REAP方法
        '''
        percentile=0.1
        num_imgs=bg_imgs.shape[0]
        coeffs = torch.zeros(num_imgs, 2).cuda()
        for i in range(num_imgs):
            if obj_masks[i].sum()<=0:
                continue
            real_pixels=torch.masked_select(bg_imgs[i], obj_masks[i].to(torch.bool))
            min_ = torch.quantile(real_pixels, percentile)
            max_ = torch.quantile(real_pixels, 1 - percentile)
            coeffs[i] = torch.tensor([max_ - min_, min_], dtype=torch.float32).reshape(-1,2)
        relighted_sign_imgs=sign_imgs*coeffs[:,0][:,None,None,None]+coeffs[:,1][:,None,None,None]
    return relighted_sign_imgs.clamp(0,1)

def random_overlay(imgs, patch, patch_mask, patch_transforms,bright,b_relight=False):    
    batch_size=imgs.shape[0]              
    dsize=[imgs.shape[-2],imgs.shape[-1]]
    patch_masks=[]
    padded_patchs=[]

    patch_masks = kornia.geometry.warp_perspective(patch_mask, patch_transforms, dsize,'bilinear')
    padded_patchs = kornia.geometry.warp_perspective(patch, patch_transforms, dsize,'bilinear')
    if b_relight:
        relighted_patch=relight(imgs,padded_patchs,patch_masks,'my')
    else:
        relighted_patch=padded_patchs
    relighted_patch=adjust_brightness(relighted_patch, bright, True)
    inverted_mask = (1-patch_masks)
    return  relighted_patch * patch_masks + imgs * inverted_mask# +white * white_masks

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False
    
def iou(box: torch.Tensor, boxes: torch.Tensor):
    """ 计算一个边界框和多个边界框的交并比

    Parameters
    ----------
    box: Tensor of shape `(4, )`
        一个边界框

    boxes: Tensor of shape `(n, 4)`
        多个边界框

    Returns
    -------
    iou: Tensor of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = torch.min(boxes[:, 2:], box[2:])
    xy_min = torch.max(boxes[:, :2], box[:2])
    inter = torch.clamp(xy_max-xy_min, min=0)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    return inter/(area_box+area_boxes-inter)

def mbbox(bbox_pred,bbox_real):
    # ious=iou(bbox_real.reshape(4,-1),bbox_pred)
    ious=bbox_tc.box_iou(bbox_real,bbox_pred)
    # _,idx=torch.topk(ious,topk)
    # bbox_ret=bbox_pred(idx)
    return ious>0.5#idx[:topk],ious[0,idx[:topk]]

def splits(gts,preds,cls_scores,label,conf_thresh,flag_loss):
    assert len(gts)==len(preds)
    ps=[]
    vs=[]
    ious=[]
    criterion = nn.CrossEntropyLoss(reduce=False)  
    labels = torch.tensor([0], dtype=torch.long).cuda()
    const_k=0.5
    tlab=Variable(torch.from_numpy(np.eye(num_classes)[label]).cuda().reshape(1,-1).float())
    empty=torch.tensor([0],dtype=torch.float32).cuda()
    for i in range(len(gts)):
        gt_tmp=gts[i]
        pred_tmp=preds[i]
        cls_scores_tmp=cls_scores[i]
        assert(len(pred_tmp)==len(cls_scores_tmp))
        choosed_confs=torch.softmax(cls_scores_tmp, dim=-1)
        labeled_elements = choosed_confs[:, label]  

        # iou loss
        # ious=bbox_tc.box_iou(pred_tmp,gt_tmp)
        # idx_iou=ious>0.25
        # ious=ious*idx_iou
        # ious=ious.reshape(-1)
        ious=bbox_tc.box_iou(pred_tmp,gt_tmp)
        ious=torch.exp(0.01*(ious**2))-1
        ious=ious.reshape(-1)

        # confs loss
        masked_matrix = choosed_confs.clone()  
        masked_matrix[:, label] = -float('inf') 
        sorted_matrix, _ = torch.sort(masked_matrix, dim=1, descending=True)   # 对修改后的每一行进行降序排序  
        first_largest  = sorted_matrix[:, 0]  # 选择每一行的第二个元素，即除label元素外的第二大元素  
        confs = torch.clamp(labeled_elements - first_largest+const_k,min=0)         # 计算差值  

        if 0==flag_loss:
            focal_confs=ious*confs
        elif 1==flag_loss:
            focal_confs=ious*labeled_elements.mean()
        elif 2==flag_loss:
            focal_confs=confs
        elif 3==flag_loss:
            focal_confs=labeled_elements.mean()        
        elif 4==flag_loss:
            focal_confs=labeled_elements.max()    
        else:
            raise ValueError('Not Implement')


        vs.append(focal_confs)
    return vs


def calculate_frequencies(vector, K, img_size):  
    # 检查输入向量是否所有值都在[0, 1]范围内  
    if not all(0 <= x <= 1 for x in vector):  
        raise ValueError("Vector elements must be in the range [0, 1].")  
      
    # 初始化频率和区间字典  
    frequencies = {}  
      
    # 计算前K-1个子区间的结束点，并确保最后一个区间的结束点是img_size  
    intervals = np.linspace(0, img_size, K, endpoint=False, dtype=int)  
    intervals = np.concatenate((intervals, [img_size]))  # 确保最后一个区间结束于img_size  
      
    # 遍历每个子区间，初始化频率为0，并构造区间字符串  
    for i in range(K):  
        start_pixel = intervals[i]  
        end_pixel = intervals[i + 1]  
        range_str = f'[{start_pixel}, {end_pixel})'  # 使用半开半闭区间表示法  
        frequencies[range_str] = 0  
      
    # 遍历向量中的每个元素，统计每个子区间中元素的出现次数  
    for value in vector:  
        pixel_value = int(value * img_size)  
        for range_str, count in frequencies.items():  
            range_start, range_end = map(int, range_str[1:-1].split(', '))  
            if range_start <= pixel_value < range_end:  
                frequencies[range_str] += 1  
                break  
      
    # 计算频率  
    total_elements = len(vector)  
    frequencies = {k: v / total_elements for k, v in frequencies.items()}  
      
    return frequencies  

def sample_from_frequencies(frequencies, batch_size):  
    # 解析frequencies字典，获取区间和对应的频率  
    intervals = list(frequencies.keys())  
    probs = list(frequencies.values())  
      
    # 采样得到的整数集合  
    sampled_integers = []  
      
    # 进行batch_size次采样  
    for _ in range(batch_size):  
        # 根据频率采样一个子区间  
        chosen_interval = np.random.choice(intervals, p=probs)  
          
        # 解析选择的区间，获取start_pixel和end_pixel  
        start_pixel, end_pixel = map(int, chosen_interval[1:-1].split(', '))  
        if chosen_interval[-1] == ')':  # 如果是半开半闭区间[start, end)  
            end_pixel -= 1  # 调整end_pixel以符合半开半闭区间的定义  
          
        # 在start_pixel和end_pixel之间随机选择一个整数  
        sampled_integer = random.randint(max(1,start_pixel), end_pixel)  
          
        # 将随机产生的整数添加到集合中  
        sampled_integers.append(sampled_integer)  
      
    return np.vstack(sampled_integers)

def adjust_probabilities(frequencies, iterations, iterations_max):  
    # 计算调整系数  
    adjustment_factor = np.cos(iterations / iterations_max * np.pi / 2)
  
    # 调整每个子区间的概率  
    adjusted_frequencies = {}  
    total_adjusted_prob = 0  
    for interval, prob in frequencies.items():  
        adjusted_prob = np.power(prob , 1-adjustment_factor)  
        total_adjusted_prob += adjusted_prob  
        adjusted_frequencies[interval] = adjusted_prob  
  
    # 归一化概率  
    normalized_frequencies = {interval: prob / total_adjusted_prob for interval, prob in adjusted_frequencies.items()}  
  
    return normalized_frequencies 

def color_convert(model_color,img,color_converter):
    if 'None'==color_converter:
        return img
    elif 'MLP'==color_converter:
        img=img.permute(0,2,3,1)
        img_tensor = img.reshape((-1,3))  # 添加batch维度  
        prediction = model_color(img_tensor.clip(0,1))
        # prediction = (1+0.05*torch.randn_like(prediction))*prediction
        prediction = prediction.reshape(img.shape)
        return prediction.permute(0,3,1,2) 
    elif 'QUA'==color_converter:
        img=img.permute(0,2,3,1)
        img_tensor = img.reshape((-1,3))  # 添加batch维度  
        prediction = model_color(img_tensor.clip(0,1))
        prediction = prediction.reshape(img.shape)
        return prediction.permute(0,3,1,2)
    elif 'NPS'==color_converter:
        return img
    else:
        raise ValueError('Not Implement')
    

def load_color_model(model_name):
    if 'None'==model_name:
        model_color=None
    elif 'MLP'==model_name:
        from print_calib.print_pred_mlp import MLP 
        model_color=MLP(3, 32, 3).cuda()
        model_color.load_state_dict(torch.load('./print_calib/results/mlp_model.pth', map_location=torch.device("cuda")))
    elif 'QUA'==model_name:
        from print_calib.print_pred_qua import QUA 
        model_color=QUA().cuda()
        model_color.load_state_dict(torch.load('./print_calib/results/qua_model.pth', map_location=torch.device("cuda")))
    elif 'NPS'==model_name:
        model_color=None
    else:
        raise ValueError('Not Implement')
    return model_color

if __name__ == '__main__':

    '''
    超参数设置
    '''
    if len(sys.argv)!=7:
        print('Manual Mode !!!')
        model_name='YOLOV2_COCO'
        attack_type='my'
        sign_name='stop_local' #M1-4 W1-1 W1-2
        
        flag_with_sample=0       # 0 使用渐进采样 1 均匀采样 2 直接使用目标采样 3 使用COCO分布
        flag_loss=0              # 0 iou*conf  1 iou*mean 2 conf 3 mean 4 max
        flag_relight=True       # 是否使用光照调整
    else:
        print('Terminal Mode !!!')
        model_name=sys.argv[1]
        attack_type=sys.argv[2]
        sign_name=sys.argv[3]

        flag_with_sample=int(sys.argv[4])
        flag_loss=int(sys.argv[5])                 
        flag_relight=sys.argv[6]=='True'
        
    assert(attack_type in ['my'])


    if 'YOLOV2_COCO'==model_name:
        # 模型初始化
        checkpoint_file = './yolov2/weights/detection/coco/yolov2_d19/yolov2_d19_26.6_46.0.pth'
        input_size=416
        num_classes=80
        cfg = yolo_config.yolov2_d19_cfg
        model=yolo_net(device=torch.device("cuda"), 
                   input_size=input_size, 
                   num_classes=80, 
                   trainable=False, 
                   conf_thresh=0.3,
                   nms_thresh=0.5,
                   anchor_size=cfg['anchor_size_coco'])
        model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device("cuda")))
        model.name=model_name
        model.cuda().eval()
        model.num_classes=num_classes

        # 攻击参数初始化
        psize = 416 
        iterations=2000
        lr=3e-2
        TARGET_LABELS=11  
        batch_size=20   
        weight_l2=0#2e-5
        weight_nps=0#0.01
        max_rotation=20 
        b_offset=10/255 
        # flag_relight=False
        hostname = socket.gethostname()  
        if 'dell-4090-1'==hostname:
            dir_background='/media/ExtHDD2/Dataset/COCO_2017/unziped/train2017/'
        elif 'dell-Precision-5820-Tower-X-Series'==hostname:
            dir_background='/home/Datasets/COCO/train2017/'
        else:
            dir_background='/ssdfs/datahome/tj18550015/GraduateDataset/COCO_2017/unziped/train2017/'
        dirstribution=np.loadtxt('distribution_COCO.txt').astype(np.float32)
        if 3==flag_with_sample:
            dirstribution=np.loadtxt('distribution_TT100K.txt').astype(np.float32)
        conf_thresh=0.30

    elif 'YOLOV8_TT100K'==model_name:
        # 模型初始化
        checkpoint_file = './yolo_rt/weights/tt100k/yolov8_l/yolov8_l_best.pth'
        input_size=416
        num_classes=45
        args_y8=args_test()
        model_cfg = build_model_config(args_y8)
        model = build_model(args_y8, model_cfg, torch.device("cuda"), num_classes, False)
        model = load_weight(model, checkpoint_file, False)
        model.cuda().eval()
        model.name=model_name
        hostname = socket.gethostname()  
        if 'dell-4090-1'==hostname:
            dir_background='/media/ExtHDD2/Dataset/GraduateDataset/TT100KD100/images/train/'
        elif 'dell-Precision-5820-Tower-X-Series'==hostname:
            dir_background='/home/Datasets/TT100K/TT100KD100/images/train/'
        else:
            dir_background='/ssdfs/datahome/tj18550015/GraduateDataset/TT100KD100/images/train/'
        conf_thresh=0.01

        if 'p11' in sign_name: TARGET_LABELS=6
        elif 'i4' in sign_name: TARGET_LABELS=12
        elif 'pne' in sign_name: TARGET_LABELS=14
        elif 'pl30' in sign_name: TARGET_LABELS=21
        elif 'w55' in sign_name: TARGET_LABELS=24
        elif 'p26' in sign_name: TARGET_LABELS=25
        elif 'i2' in sign_name: TARGET_LABELS=31
        elif 'w57' in sign_name: TARGET_LABELS=36
        elif 'w59' in sign_name: TARGET_LABELS=38
        elif 'i5' in sign_name: TARGET_LABELS=42
        else: raise ValueError("Not Implement")

        # 攻击参数初始化
        psize = 416 
        iterations=2000
        lr=3e-2
        batch_size=20 
        weight_l2=0#2e-5
        weight_nps=0
        weight_tv=0
        max_rotation=20 
        b_offset=10/255 
        # flag_relight=True
        dirstribution=np.loadtxt('distribution_TT100K.txt').astype(np.float32)
        if 3==flag_with_sample:
            dirstribution=np.loadtxt('distribution_COCO.txt').astype(np.float32)
    else:
        raise ValueError('Not Implement')
    saved_name=model_name+'_'+sign_name+'_'+attack_type
    saved_dir_att=os.path.join('./AEs',model_name,sign_name,'Samp{}_Loss{}_L{}'.format(flag_with_sample,flag_loss,flag_relight))
    if not os.path.exists(saved_dir_att): 
        try:
            os.makedirs(saved_dir_att) 
        except:
            print('Exists')
    saved_dir_writer=os.path.join(saved_dir_att,attack_type)
    if not os.path.exists(saved_dir_writer): os.makedirs(saved_dir_writer)
    writer = SummaryWriter(saved_dir_writer)

    '''
    图像初始化
    '''

    bg_imgs=torch.ones((input_size,input_size),dtype=torch.float32)
    img={}
    img['img']=bg_imgs
    if 'YOLOV2_COCO' in model_name:
        img['img_metas']={'mean':torch.tensor((0.485,0.456,0.406),dtype=torch.float32).cuda(),
                        'std':torch.tensor((0.229,0.224,0.225),dtype=torch.float32).cuda()}
    bg_imgs_list = list(filter(file_filter, os.listdir(dir_background)))


    '''
    掩码初始化
    '''
    sign_dir='./other_signs'
    MASK_CHANGE = np.array(PIL.Image.open(os.path.join(sign_dir,sign_name+'_change.png')).convert("L").resize((psize, psize)))
    MASK_CHANGE = torch.from_numpy(MASK_CHANGE).to(torch.float32).cuda().unsqueeze(0).unsqueeze(0)/255.
    MASK_ALL=np.array(PIL.Image.open(os.path.join(sign_dir,sign_name.replace('_local','')+'_whole.png')).convert("L").resize((psize, psize)))
    MASK_ALL=torch.from_numpy(MASK_ALL).to(torch.float32).cuda().unsqueeze(0).unsqueeze(0)/255.
    GT=bbox_generator(torch.Tensor([0]),torch.Tensor([0]),torch.Tensor([psize]),torch.Tensor([psize])).unsqueeze(0)
    GT=GT.repeat((batch_size,1,1,1)).cuda()

    standard_sign=torch.from_numpy(np.array(PIL.Image.open(os.path.join(sign_dir,sign_name.replace('_local','')+'.png')).convert("RGB").resize((psize, psize)))).to(torch.float32).cuda().unsqueeze(0)/255.
    standard_sign=standard_sign.permute(0,3,1,2).clip(0,1)

    '''
    patch初始化
    '''
    patch=standard_sign.clone().clip(0.01,0.99)
    patch=torch.atanh(2*patch-1)
    patch.requires_grad=True

    '''
    采样概率初始化
    '''
    K = 50  # 将[0, img_size]区间均匀分割成5个子区间  
    frequencies = calculate_frequencies(dirstribution[:,-1], K, input_size)  


    '''
    优化
    '''
    MSELoss = torch.nn.MSELoss(reduction='mean')
    printfile='non_printability/30values.txt'
    nps_calculator = NPSCalculator(printfile, psize).cuda()
    optimizer=torch.optim.Adam([patch],lr=lr)
    # optimizer=torch.optim.SGD([patch],lr=lr)

    if 'sysadv'==attack_type:
        # 根据SLE生成分布
        pixels = range(30,100)
        probs = np.array([1/(pix**2) for pix in pixels])
        probs = probs/probs.sum()
        distrib = rv_discrete(values=(range(len(pixels)), probs))

    # iterations=5000
    pbar = tqdm(range(iterations))
    for i_pbar in pbar:
        backgrounds=[]
        for i in range(batch_size):
            background=PIL.Image.open(os.path.join(dir_background,choice(bg_imgs_list))).resize((input_size, input_size)).convert('RGB')
            backgrounds.append(torch.from_numpy(np.array(background)).unsqueeze(0).permute(0,3,1,2).cuda().to(torch.float32)/255)
        bg_imgs=torch.vstack(backgrounds)
        dist=np.vstack([choice(dirstribution) for i in range(batch_size)])

        '''
        增强参数初始化
        '''
        if 0==flag_with_sample:
            adjusted_frequencies = adjust_probabilities(frequencies, i_pbar, iterations) 
        elif 1==flag_with_sample: 
            adjusted_frequencies = adjust_probabilities(frequencies, 0, iterations) 
        elif 2==flag_with_sample: 
            adjusted_frequencies = adjust_probabilities(frequencies, iterations, iterations)
        elif 3==flag_with_sample: 
            adjusted_frequencies = adjust_probabilities(frequencies, i_pbar, iterations)
        else:
            raise ValueError('Not Implement')
        scales_med = torch.from_numpy(sample_from_frequencies(adjusted_frequencies, batch_size)).cuda()/input_size
        # print(scales_med.mean())
        border_left=(1-scales_med)/2
        trans_med=torch.rand((batch_size,2)).cuda()
        trans_med=-1*border_left+2*border_left*trans_med
        trans_med=trans_med*input_size
        rots_med=torch.tensor(np.random.uniform(-1,1,size=batch_size).astype(np.float32)).cuda()
        b_rnds=torch.tensor(np.random.uniform(-1*b_offset,b_offset,size=batch_size).astype(np.float32)).cuda()
        centers=torch.Tensor([[psize/2,psize/2]]).repeat((batch_size,1)).cuda()
    
        # 图像
        patch_img_add=(torch.tanh(patch) + 1)/2
        patch_img_add=patch_img_add*MASK_CHANGE+standard_sign*(1-MASK_CHANGE)*MASK_ALL
        showarray(patch_img_add[0].clip(0,1).detach().cpu(),'patch')

        patch_transformations=kornia.geometry.get_affine_matrix2d(translations=trans_med,
                                                center=centers,
                                                scale=torch.hstack([scales_med,scales_med]),
                                                angle=rots_med*max_rotation) 
        
        img['img']=random_overlay(bg_imgs.detach(),patch_img_add.repeat(batch_size,1,1,1), MASK_ALL.repeat(batch_size,1,1,1),patch_transformations,b_rnds,flag_relight)
        showarray(img['img'][0].clip(0,1).detach().cpu(),'patch_with_bg_my')
        gt_transformed=generate_gt(GT,patch_transformations,psize)
        gt_transformed=[tmp/input_size for tmp in gt_transformed]
        vs, second_stage_loc_bboxes=inference(model,img, gt_transformed, TARGET_LABELS,conf_thresh,flag_loss)
        second_stage_cls_scores=torch.vstack([v_tmp.sum() for v_tmp in vs]).mean()

        # 计算损失函数
        tv_loss=total_variation(patch_img_add,'mean').mean()
        l2=torch.norm(standard_sign*MASK_ALL-patch_img_add.clip(0,1)*MASK_ALL)  # nae3 89.2
        nps_loss=nps_calculator(patch_img_add)

        gt_transformed=generate_gt(GT,patch_transformations,psize)
        loss = second_stage_cls_scores + weight_l2*l2+weight_nps*nps_loss

        
        # 迭代优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        # 计算mAP
        gt_for_eval=[{'bboxes':bbox.detach().cpu().numpy().reshape(-1,4),'labels':np.array(TARGET_LABELS).reshape(-1)} for bbox in gt_transformed]
        pbar.set_description('[{}_{}][{}/{}] cls:{:.4f} l2:{:.2f} tv:{:.2f} nps:{:.2f}'.format(model_name,attack_type,i_pbar,iterations,second_stage_cls_scores,l2,tv_loss,nps_loss))
    
        writer.add_scalar('loss', loss,i_pbar)
        writer.add_scalar('loss_cls', second_stage_cls_scores,i_pbar)
        writer.add_scalar('loss_l2', l2,i_pbar)
        writer.add_scalar('loss_tv', tv_loss,i_pbar)
        writer.add_scalar('loss_nps', nps_loss,i_pbar)

        saved_patch=patch_img_add[0].clip(0,1).detach().cpu()
        saved_patch=saved_patch*MASK_ALL[0].cpu()+torch.ones_like(saved_patch)*(1-MASK_ALL[0].cpu())
        showarray(saved_patch,saved_dir_att+'/{}'.format(saved_name))