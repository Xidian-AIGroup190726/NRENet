import warnings

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
import torch.nn.functional as F
import math
from PIL import Image
from mmdet.core import multi_apply, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
import numpy as np
import random
from skimage.metrics import structural_similarity as compare_ssim
from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt 
import cv2

INF = 1e8
global weights
weights=[]

@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        self.dimension = 1  # 变量个数
        self.time = 10  # 迭代的代数
        self.size = 62  # 种群大小
        self.bound = []  # 变量的约束范围
        #low = [48]
        #up = [96]
        #self.bound.append(low)
        #self.bound.append(up)
        self.v_low = -2
        self.v_high = 2
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]
        # self.loss_features = loss_features
        global weights
        weights=[]
        global iters
        iters = []
        global ms
        ms = []
        ms.append(48)
        global temp
        temp=[]
        temp.append(100000)
        global r
        r =[]
        global r1
        r1 = []
        global loss_s
        loss_s = []
        global final_best
        final_best = []
        final_best.append(64)
        global LOSS
        LOSS=[]
        global zz
        zz=[]
        global epoch
        epoch = []
        epoch.append(0)
        global x_iter
        global ap
        ap=[]
        x_iter=[]
        x_iter.append(48)
        global small_target
        small_target =[]
        global num
        num = []
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_features = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        #print(feats.shape)
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        #print("shape:",x.shape)#torch.Size([8, 256, 52, 64])
        #print("done",(x.shape)[0])
        #print("before:",len(weights))
        if len(weights)>=5:
            weights.clear()
            ########HRSID 455 SSDD 101
        if len(iters)%101 == 0 and len(iters)!=0 and len(ap)==0:
          epoch.append(0)
        ap.append(0)
        if len(ap)>=5:
          ap.clear()
        #print("epoch:",len(epoch),len(iters),len(ap))
        #print("x",x[0][255])
        #savepath = r'/media/ExtDisk/yxt/ture map/'
        #self.draw_features(8, 8, x[0][255].cpu().detach().numpy(), "{}/f1_conv1.png".format(savepath))
        if len(epoch)%2==0:# or len(epoch)==1:# or len(epoch)>=51 and len(iters)%102!=0:
          wmap = self.conv_features(x)
          weight = torch.sigmoid(wmap)
          weights.append(weight)
          for i in range((x.shape)[0]):
             x[i].data *=weight[i]
        ##########
          #print("weightSSS:",len(weights),weight.shape)
          #print("weight",weight[0])
        #########
        #if (0<len(iters)<=4000000  ) and len(ap)==1:
         # num.append(x)
          #print(num[-1].shape)
        
         
        
        
        #print("weightss",weight[7])
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        #if len(ap)==3:
         # print("cls:",cls_score[0].sigmoid())
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # for i in range(len(weights)):
        # print("weight:",weights[i].shape)
        # weights is a list of 5 
        # print("done")
        # print(weights[0].shape)#torch.Size([8, 1, 60, 64])
        #for i in range(len(weights)):
          #print("weight:",weights[i].shape,len(weights))
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        #print(featmap_sizes)
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        #print((all_level_points[0]))
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        
        #target1 = F.one_hot(labels[3], num_classes=2)
        
        #print("target1:",target1)
        #u=0
        #for i in range(len(target1)):
         #   if target1[i]>0:
          #      u=u+1
        #print("tar:",len(target1),u)

        # print(len(img_metas),((img_metas[7])['img_shape'])[0])
        img_shapes = []
        for i in range(len(img_metas)):
            img_shapes.append((img_metas[i])['img_shape'])
        
        name = []
        for i in range(0,len(gt_bboxes)):
            name.append(img_metas[i]['ori_filename'])
        res = self.cal_res(name)# caculate shang
        i=0
        flag=0
        #print("iters:",len(iters))
        pos_=0
        neg_=0
        if 10000000<len(iters)<=100000000:
          for j in range(8):
            if res[j]>6:
              flag+=1
              while i<=255:
                b=((num[-1][j][i]).cpu()).detach().numpy()
                #print(i,b.sum(),name[j],b)
                if b.sum()>=0:
                  pos_+=1
                else:
                  neg_+=1
                #plt.matshow(b,cmap=plt.cm.Reds)
                #plt.title(name[j])
                #plt.savefig('/media/ExtDisk/yxt/map/ma/'+str(i))
                i+=1
                #print("pg:",pos_,neg_,pos_/(neg_+0.000000001))
            if flag!=0:
              print(name[j])
              break
        #print("pgg:",pos_,neg_,pos_/(neg_+0.000000001))
        i=0
        flag=0
        if 102999<len(iters)<=103004:
          for j in range(8):
            if res[j]>6:
              flag+=1
              while i<=255:
                b=((num[-1][j][i]).cpu()).detach().numpy()
                print(i,b.sum(),name[j],b)
                plt.matshow(b,cmap=plt.cm.Reds)
                plt.title(name[j])
                plt.savefig('/media/ExtDisk/yxt/map/map_3/'+str(i))
                i+=1
            if flag!=0:
              break

        i=0
        flag=0
        
        if 105300<len(iters)<=105306:
          for j in range(8):
            if res[j]>6:
              flag+=1
              while i<=255:
                b=((num[-1][j][i]).cpu()).detach().numpy()
                print(i,b.sum(),name[j],b)
                plt.matshow(b,cmap=plt.cm.Reds)
                plt.title(name[j])
                plt.savefig('/media/ExtDisk/yxt/map/map_5/'+str(i))
                i+=1
            if flag!=0:
              break
        
        area_s = []
        WH = []
        for k in range(len(gt_bboxes)):
            mian = []
            wh=[]
            #print(len(gt_bboxes[k]))
            for h in range(len(gt_bboxes[k])):
                wh.append(max(((gt_bboxes[k][h])[2]-(gt_bboxes[k][h])[0])/2,((gt_bboxes[k][h])[3]-(gt_bboxes[k][h])[1]))/2)
                mian.append(((gt_bboxes[k][h])[2]-(gt_bboxes[k][h])[0])*((gt_bboxes[k][h])[3]-(gt_bboxes[k][h])[1]))
            area_s.append(mian)
            small_target.append(mian)
            WH.append(wh)
        rmin=[]
        rmax=[]
        for i in range(len(area_s)):
            p=0
            for j in range(len(area_s[i])):
              if res[i]>=6 or area_s[i][j]<=(32*32):
                  p=p+1
            if p!=0:
              index_min = area_s[i].index(min(area_s[i]))
              index_max = area_s[i].index(max(area_s[i]))
              rmin.append(WH[i][index_min])
              rmax.append(WH[i][index_max])
        rsumin=0
        rsumax=0
        for i in range(len(rmin)):
            rsumin = rsumin+rmin[i]*2 #per image's min ship's max width
            rsumax = rsumax+rmax[i]*2 #per image's max ship's max width
        if len(rmin)!=0:
          Rmin = rsumin/len(rmin)
        else:
            Rmin=0
        if len(rmax)!=0:
          Rmax = rsumax/len(rmax)
        else:
            Rmax=0
        #print(name)
        ####print(Rmin,Rmax)
        low = [Rmin]
        up = [Rmax]
        self.bound.append(low)
        self.bound.append(up)

        #res = self.cal_res(name)
        #print(res)
        iters.append(0)
        if len(epoch)%2 == 0:#or len(epoch)==1:# or len(epoch)>=51:

          feature_target = self.feature_map_target(featmap_sizes, gt_bboxes, all_level_points, img_shapes,res)
          #for j in range(8):
          

          #print("weights:",(weights[3])[0])
          #print("target:",(feature_target[2])[0])

          feature_loss = self.loss_features(weights, feature_target)
        
          


          
        weights.clear()
        # print(len(gt_bboxes),gt_bboxes[7])
        # gt_bboxes is a list of 8 images' bbox lists
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        #print("label:",labels)
        
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        

        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        #print("cls:",num_pos,flatten_cls_scores,flatten_labels)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()



        if len(epoch)%2== 0:# or len(epoch)==1:
          loss_sum = 0.8*loss_cls + 0.8*loss_bbox + 0.2*feature_loss + 0.2*loss_centerness
          loss_s.append(loss_sum)
          #print("loss_sum:",loss_sum)
          loss10q = []
          loss10h = []#10 600
          a = 10
          ##ssdd 612 hrsid 2730
          if (len(r)%a==0 or len(r)==0) and len(r)<2730:
          #loss_sum = 0.4*loss_cls + 0.4*loss_bbox + 0.1*feature_loss + 0.1*loss_centerness
          #loss_s.append(loss_sum)
            if len(r)==0:
              loss10q.append(loss_s[0])
              loss10h.append(loss_s[0])
            elif len(r)>a:
              for i in range(a):
                loss10h.append(loss_s[-i-1])
                loss10q.append(loss_s[-i-a])
            else:
              for i in range(a):
                loss10h.append(loss_s[-i-1])
              loss10q.append(loss_s[0])
            print("q:",min(loss10q))
            print("h:",min(loss10h))

            #self.x[0][0] = random.uniform(self.bound[0][0], self.bound[1][0])
            #self.v[0][0] = random.uniform(self.v_low, self.v_high)
            #self.p_best[0] = self.x[0]  # 储存最优的个体
            #heyibufenxiedao loss nali
            # 做出修改
            if len(r1)==0:
              self.x[0][0] = random.uniform(self.bound[0][0], self.bound[1][0])
              self.v[0][0] = random.uniform(self.v_low, self.v_high)
              self.p_best[0] = self.x[0]  # 储存最优的个体
              self.g_best = self.p_best[0]
            print("u",self.x[0][0])
            for gen in range(self.time):
              self.update(self.size)
            u=self.x[0][0]
            x_iter.append(u)
            #print("x",x_iter)
            if len(r)==0 or len(r)==1:
              (self.p_best[0])[0] = x_iter[-1]
            elif len(r)>1:
                if min(loss10q)>=min(loss10h):
                    (self.p_best[0]) = x_iter[-2]
                else:
                    (self.p_best[0]) = x_iter[-3]
            #self.p_best[0] = self.x[0]
            #self.p_best[i] = self.x[i]
            #ms.append((self.p_best[-1])[0])
            ##print(self.p_best)
            
            LOSS.append([min(loss10h),ms[-1]])
            #ms.append((self.p_best[0])[0])
            ms.append(self.x[0][0])
            ####print("LOSS:",LOSS,ms)
            a=[]
            for i in range(len(LOSS)):
              a.append(LOSS[i][0])
            index_min = a.index(min(a))
            self.g_best = LOSS[index_min][1]
            r1.append(0)
          r.append(0)
          ####612 2730
        if len(r)==2730:
          a=[]
          for i in range(len(LOSS)):
              a.append(LOSS[i][0])
          index_min = a.index(min(a))
          self.g_best = LOSS[index_min][1]
          ms.append(self.g_best)
        ##print("ms:",len(r),ms[-1])
        if len(epoch)%2 == 0:# or len(epoch)==1:# or len(epoch)>=51:
            return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,#,
            loss_centerness=loss_centerness,
            loss_features=feature_loss)
        else:
          return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    #def interval_confirmation(self):

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
        
        
    def draw_features(self,width, height, x, savename):
      
      fig = plt.figure(figsize=(16, 16))
      fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
      for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        #plt.imshow(img)
        print("{}/{}".format(i, width * height))
      fig.savefig(savename, dpi=100)
      fig.clf()
      plt.close()

    def feature_map_target(self, featmap_sizes, gt_bboxes, all_level_points, img_shapes,res):
        stride1 = [8,16,32, 64, 128]
        feature_target = []
        cls_target = []
        
        for i in range(len(featmap_sizes)):
            save_feature_weight = []
            save_cls_weight = []
            for j in range(len(gt_bboxes)):
                p = -1
                small = 0
                for e in range(len(small_target[j])):
                  #print(small_target[j][e])
                  if small_target[j][e]<=1024:
                    small+=1
                  #print(small)
                
                if res[j]>=6:# or small!=0:
                  if i==0:
                    weights = torch.zeros(featmap_sizes[i+1][0], featmap_sizes[i+1][1], dtype=torch.float32,
                                      device=torch.device('cuda:1'))
                    
                  else:
                    weights = torch.zeros(featmap_sizes[i][0], featmap_sizes[i][1], dtype=torch.float32,
                                      device=torch.device('cuda:1'))
                  
                  for k in range(weights.shape[0]):
                      for h in range(weights.shape[1]):
                         p = p + 1
                         locations = (all_level_points[i])[p]
                         sizes = (img_shapes[j][0], img_shapes[j][1])  # 306,500
                         #print("loca:",locations)
                         xl_yl = locations - (stride1[i] / 2)
                         xr_yr = locations + (stride1[i] / 2)
                         box1 = [xl_yl[0], xl_yl[1], xr_yr[0], xr_yr[1]]
                         for l in range(len(gt_bboxes[j])):
                           box2 = (gt_bboxes[j])[l]
                           l1 = 1 if box2[2]>=locations[0]>=box2[0] else 0
                           r1 = 1 if box2[3]>=locations[1]>=box2[1] else 0
                           in_or_out = min(l1,r1)
                           #print("in_or_out:",locations,box2,in_or_out)
                           range_box = max(box2[2]-box2[0],box2[3]-box2[1])
                           m =ms[-1] #if (box2[2]-box2[0])*(box2[3]-box2[1])<=1024 else 24
                           xl = box2[0]-m if (box2[0]-m)>=0 else 0
                           yl = box2[1]-m if (box2[1]-m)>=0 else 0
                           xr = box2[2]+m if (box2[2]+m)<=sizes[1]-1 else sizes[1]-1
                           yr = box2[3]+m if (box2[3]+m)<=sizes[0]-1 else sizes[0]-1
                           box22 = [xl,yl,xr,yr]
                          #print("box22:",box22)
                           
                           if i == 0 and range_box<=64:
                             if IOU(box1, box22) != 0:
                               weights[k][h] += 1

                           
                             if IOU(box1, box2) != 0:
                                weights[k][h] += 2
                                


                           if i==1 and range_box>64 and range_box<=128:
                             if IOU(box1, box22) != 0:
                               weights[k][h] += 1
                           
                             if IOU(box1, box2) != 0:
                                weights[k][h] += 2
                                

                           if i==2 and range_box>128 and range_box<=256:
                             if IOU(box1, box22) != 0:
                               weights[k][h] += 1
                          
                             if IOU(box1, box2) != 0:
                                weights[k][h] += 2
                                

                           if i==3 and range_box>256 and range_box<=512:
                             if IOU(box1, box22) != 0:
                               weights[k][h] += 1
        
                             if IOU(box1, box2) != 0:
                                weights[k][h] += 2
                      
                         weights[k][h] = torch.sigmoid(weights[k][h])


                else:
                  if i==0:
                    weights = torch.ones(featmap_sizes[i+1][0], featmap_sizes[i+1][1], dtype=torch.float32,
                                      device=torch.device('cuda:1'))
                  else:
                    weights = torch.ones(featmap_sizes[i][0], featmap_sizes[i][1], dtype=torch.float32,
                                      device=torch.device('cuda:1'))
                     
                if i==0:
                  w = weights.cpu().detach().numpy()
                  img = Image.fromarray(w)
                  d = img.resize((featmap_sizes[i][1], featmap_sizes[i][0]))
                  ww = np.array(d)
                  weights = torch.tensor(ww,dtype=torch.float32,
                                      device=torch.device('cuda:1'))
                  
                save_feature_weight.append(weights)
                
            feature_target.append(save_feature_weight)
            
        #print(feature_target[0][7].shape,len(feature_target))
        small_target.clear()
        return feature_target

    def cal_res(self,img_name):
        res1 = []
        #print(len(img_name),img_name[0])
        for j in range(len(img_name)):
          tmp = []
          for i in range(256):
              tmp.append(0)
          val = 0
          k = 0
          res = 0
          #I1 = Image.open('.../train_image/'+str(img_name[j]))
          I = Image.open('.../ssdd_coco/train/train_image/'+str(img_name[j]))
          #I = I1.resize((400, 400))
          greyIm=I.convert('L')
          img=np.array(greyIm)
          for i in range(len(img)):
              for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k =  float(k + 1)
          for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
          tmp = np.array(tmp)
          for i in range(len(tmp)):
            if(tmp[i] == 0):
                res=res
            else:
                res=float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
          res1.append(res)
        return res1

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        # print(points[0]) 
        num_levels = len(points)
        # print(num_levels) #==5
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)
        #print("label:",labels_list)
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        #print("lllll:",labels_list)
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))
        
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF

        min_area, min_area_inds = areas.min(dim=1)
        
        labels = gt_labels[min_area_inds]

        labels[min_area == INF] = self.num_classes  # set as BG
        
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                                         left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def loss_features(self, weights, feature_target):
        loss = []
        loss1 = []
        s = 0
        s1 = 0
        #print((weights[0].shape)[0])
        for i in range(len(weights)):
            for j in range((weights[0].shape)[0]):
                #print(((weights[i])[j])[0].shape,(feature_target[i])[j].shape)
                out_put = F.cosine_similarity(((weights[i])[j])[0], (feature_target[i])[j], dim=-1)
                out_put = torch.mean(out_put)
                loss.append(abs(out_put))
                #loss1.append(abs(out_put1))
        for i in range(len(loss)):
            s = s + loss[i]
        s = s / len(loss)
        s = (1-s)*5
        weights.clear()
        return s

    def loss_features1(self,weights, feature_target):
      ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
      s=0
      for i in range(len(weights)):
            for j in range((weights[0].shape)[0]):
                (feature_target[i])[j]=torch.unsqueeze((feature_target[i])[j],dim=0)
                b = torch.unsqueeze((weights[i])[j],dim=0)
                #print(b.shape)
                #(weights[i])[j] = b


      for i in range(len(weights)):
            for j in range((weights[0].shape)[0]):
                (feature_target[i])[j]=torch.unsqueeze((feature_target[i])[j],dim=0)

      for i in range(len(weights)):
        for j in range((weights[0].shape)[0]):
            b = torch.unsqueeze((weights[i])[j],dim=0)
            ssim1 = ssim(b,(feature_target[i])[j])
            print(ssim1)
      weights.clear()


def IOU(box1, box2):
    # 计算中间矩形的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    # 计算IoU
    iou = inter / (union + np.exp(-10))
    return iou
