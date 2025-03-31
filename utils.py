import argparse
import sys
sys.path.append('./')
# from detectron2.config import get_cfg
# from detectron2.engine import  default_setup
# from freesolo import add_solo_config
import os
import numpy as np
from PIL import Image, ImageColor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import re
import warnings
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.module import Module
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.overrides import (has_torch_function, handle_torch_function, has_torch_function_variadic)
from torch.nn.functional import pad, softmax, dropout
import math
import spacy
import cv2


def extract_noun_phrase(text, nlp, need_index=False):
    # text = text.lower()
    # To find phrase root
    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text

RELATION_WORDS={"left", "west",\
                "right", "east",\
                "above", "north", "top", "back", "behind",\
                "below", "south", "under", "front",\
                "bigger", "larger", \
                "closer","smaller", "tinier", "further",\
                "inside", "within", "contained",\
                "who","what","which",\
                "middle"}

def extract_nouns(text, nlp, need_index=False):
    # text = text.lower()
    # To find phrase root
    doc = nlp(text)
    noun_phrases = []
    nouns = []
    nouns_index = []
    head_noun = extract_noun_phrase(text, nlp)
    for chunk in doc.noun_chunks:
        if chunk.text==head_noun or chunk.root.text in RELATION_WORDS:
            continue
        noun_phrases.append(chunk.text)
        nouns_index.append((chunk.start,chunk.end))
        nouns.append(chunk.root.text)
    
    if need_index:
        return noun_phrases, nouns_index, nouns
    else:
        return noun_phrases, nouns
    
def extract_dir_phrase(text, nlp, need_index=False):
    # text = text.lower()
    dirflag = "none"
    diridx = 999
    deep2head = 999
    doc = nlp(text)
    for token in doc:
        if token.text == "left" and token.head.i<deep2head:
            dirflag = "left"
            diridx = token.i
            deep2head = token.head.i
        elif token.text == "right" and token.head.i<deep2head:
            dirflag = "right"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"middle","between"} and token.head.i<deep2head:
            dirflag = "middle"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"up","top","above"} and token.head.i<deep2head:
            dirflag = "up"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"down","under","bottom","low"} and token.head.i<deep2head:
            dirflag = "down"
            diridx = token.i
            deep2head = token.head.i

    if need_index:
        return dirflag, diridx
    else:
        return dirflag

def gen_dir_mask(dirflag,height,width,device):
    if dirflag=="left":
        a=torch.linspace(1,0,width)
        pmask=a.expand(height,width)
    elif dirflag=="right":
        b=torch.linspace(0,1,width)
        pmask=b.expand(height,width)
    elif dirflag=="middle":
        b1=torch.linspace(0,1,width//2)
        b2=torch.linspace(1,0,width-width//2)
        b = torch.cat([b1,b2])
        pmask=b.expand(height,width)
    # elif dirflag=="up":
    #     c=torch.linspace(1,0,height)
    #     p_up=c.expand(width,height)
    #     pmask=torch.transpose(p_up,0,1)
    # elif dirflag=="down":
    #     d=torch.linspace(0,1,height)
    #     p_down=d.expand(width,height)
    #     pmask=torch.transpose(p_down,0,1)
    else:
        pmask=torch.ones(height,width)
    
    if device:
        return pmask.to(device)
    else:
        return pmask

def generate_direction_bias(sentence, size, device): 
    h, w = size
    direction_bias_lr = torch.ones(size)
    flag_lr = False 
    if 'left' in sentence: 
        flag_lr = True 
        direction_bias_lr = (torch.arange(w, 0, -1)).expand(h, -1) 
    elif 'right' in sentence: 
        flag_lr = True 
        direction_bias_lr = (torch.arange(1, w+1, 1)).expand(h, -1)  
    direction_bias_lr = direction_bias_lr/w

    flag_tb = False
    direction_bias_tb = torch.ones(size)
    # if 'top' in sentence or 'up' in sentence: 
    #     flag_tb = True 
    #     direction_bias_tb = (torch.arange(h, 0, -1)).expand(w,-1).T
    # elif 'bottom' in sentence or 'low' in sentence: 
    #     flag_tb = True 
    #     direction_bias_tb = (torch.arange(1, 1+h, 1)).expand(w,-1).T
    
    # direction_bias_tb = direction_bias_tb/h

    if flag_tb and flag_lr:
        direction_bias = (direction_bias_lr + direction_bias_tb) /2
    else:
        if flag_lr:
                direction_bias = direction_bias_lr
        else:
            direction_bias = direction_bias_tb 
    if device:
        return direction_bias.to(device)
    else:
        return direction_bias

NULL_KEYWORDS = {"part", "image", "side", "picture", "half", "region", "section", "photo"}
LEFT_KEYWORDS = {"left", "west"}
RIGHT_KEYWORDS = {"right", "east"}
UP_KEYWORDS = {"above", "north", "top", "back", "behind"}
DOWN_KEYWORDS = {"below", "south", "under", "front"}
BIG_KEYWORDS = {"bigger", "larger", "closer"}
SMALL_KEYWORDS = {"smaller", "tinier", "further", "smallest"}
WITHIN_KEYWORDS = {"inside", "within", "contained"}

def extract_rela_word(text, nlp):
    noun_phrases, nouns = extract_nouns(text, nlp)
    if (set(nouns) & NULL_KEYWORDS):
        relaflag = "none"
    else:
        relaflag = "none"
        deep2head = 999
        doc = nlp(text)
        for token in doc:
            if token.text in LEFT_KEYWORDS and token.head.i<deep2head:
                relaflag = "left"
                deep2head = token.head.i
            elif token.text == RIGHT_KEYWORDS and token.head.i<deep2head:
                relaflag = "right"
                deep2head = token.head.i
            elif token.text in UP_KEYWORDS and token.head.i<deep2head:
                relaflag = "up"
                deep2head = token.head.i
            elif token.text in DOWN_KEYWORDS and token.head.i<deep2head:
                relaflag = "down"
                deep2head = token.head.i
            elif token.text in BIG_KEYWORDS and token.head.i<deep2head:
                relaflag = "big"
                deep2head = token.head.i
            elif token.text in SMALL_KEYWORDS and token.head.i<deep2head:
                relaflag = "small"
                deep2head = token.head.i
            elif token.text in WITHIN_KEYWORDS and token.head.i<deep2head:
                relaflag = "within"
                deep2head = token.head.i
    return relaflag

# def extract_rela_word(text, nlp):
#     noun_phrases, nouns = extract_nouns(text, nlp)
#     # print((set(nouns) & NULL_KEYWORDS))
#     if (set(nouns) & NULL_KEYWORDS):
#         relaflag = "none"
#     else:
#         relaflag = "none"
#         doc = nlp(text)
#         for token in doc:
#             if token.text in LEFT_KEYWORDS:
#                 relaflag = "left"
#             elif token.text == RIGHT_KEYWORDS:
#                 relaflag = "right"
#             elif token.text in UP_KEYWORDS:
#                 relaflag = "up"
#             elif token.text in DOWN_KEYWORDS:
#                 relaflag = "down"
#             elif token.text in BIG_KEYWORDS:
#                 relaflag = "big"
#             elif token.text in SMALL_KEYWORDS:
#                 relaflag = "small"
#             elif token.text in WITHIN_KEYWORDS:
#                 relaflag = "within"
#     return relaflag



def relation_boxes(boxi, boxj, scorei, scorej, relaword):
    # RELATIONS = [
    #     RelHeuristic(["left", "west"], lambda env: env.left_of()),
    #     RelHeuristic(["right", "east"], lambda env: env.right_of()),
    #     RelHeuristic(["above", "north", "top", "back", "behind"], lambda env: env.above()),
    #     RelHeuristic(["below", "south", "under", "front"], lambda env: env.below()),
    #     RelHeuristic(["bigger", "larger", "closer"], lambda env: env.bigger_than()),
    #     RelHeuristic(["smaller", "tinier", "further"], lambda env: env.smaller_than()),
    #     RelHeuristic(["inside", "within", "contained"], lambda env: env.within()),
    # ]
    scoreout = 0

    if relaword == "none":
        scoreout = scorei
    elif relaword == "left":
        scoreout = scorei * scorej * ((boxi[0]+boxi[2]/2)<(boxj[0]+boxj[2]/2))
    elif relaword == "right":
        scoreout = scorei * scorej * ((boxi[0]+boxi[2]/2)>(boxj[0]+boxj[2]/2))
    elif relaword == "up":        
        scoreout = scorei * scorej * ((boxi[1]+boxi[3]/2)<(boxj[1]+boxj[3]/2))
    elif relaword == "down":        
        scoreout = scorei * scorej * ((boxi[1]+boxi[3]/2)>(boxj[1]+boxj[3]/2))
    elif relaword == "big":        
        scoreout = scorei * scorej * ((boxi[2]*boxi[3])>(boxj[2]*boxj[3]))
        # scoreout = scorei * scorej * ((boxi[2]*boxi[3])/(boxj[2]*boxj[3]))
    elif relaword == "small":        
        scoreout = scorei * scorej * ((boxi[2]*boxi[3])<(boxj[2]*boxj[3]))
        # scoreout = scorei * scorej * (boxj[2]*boxj[3])/((boxi[2]*boxi[3]))
    elif relaword == "within":        
        x1 = max(boxi[0], boxj[0])
        x2 = max(x1, min(boxi[0]+boxi[2], boxj[0]+boxj[2]))
        y1 = max(boxi[1], boxj[1])
        y2 = max(y1, min(boxi[1]+boxi[3], boxj[1]+boxj[3]))
        scoreout = scorei * scorej * (x2-x1) * (y2-y1) / (boxi[2]*boxi[3])
    else :        
        scoreout = scorei

    return scoreout

def mask2img(mask_seg):
    binary_image = mask_seg.to(torch.uint8) * 255
    h, w = binary_image.shape
    output_seg = torch.zeros((h, w, 3), dtype=torch.uint8)
    output_seg[:, :, 0] = binary_image  # 红色通道
    output_seg[:, :, 1] = binary_image  # 绿色通道
    output_seg[:, :, 2] = binary_image  # 蓝色通道
    output_seg = output_seg.numpy()
    return output_seg

def mask2chw(arr):
    # Find the row and column indices where the array is 1
    (rows, cols) = np.where(arr == 1)
    # Calculate center of the mask
    center_y = int(np.mean(rows))
    center_x = int(np.mean(cols))
    # Calculate height and width of the mask
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    return (center_y, center_x), height, width


def apply_visual_prompts(
    image_array,
    mask,
    # bbox,
    visual_prompt_type=('circle',),
    visualize=False,
    color=(255, 0, 0),
    thickness=1,
    blur_strength=(15, 15),
):
    """Applies visual prompts to the image."""
    prompted_image = image_array.copy()
    if type(mask) != type(image_array):
        mask = mask.cpu().numpy()
    if 'blur' in visual_prompt_type:
        # blur the part out side the mask
        # Blur the entire image
        blurred = cv2.GaussianBlur(prompted_image.copy(), blur_strength, 0)
        # Get the sharp region using the mask
        sharp_region = cv2.bitwise_and(
            prompted_image.copy(),
            prompted_image.copy(),
            mask=np.clip(mask, 0, 255).astype(np.uint8),
        )
        # Get the blurred region using the inverted mask
        inv_mask = 1 - mask
        blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
        # Combine the sharp and blurred regions
        prompted_image = cv2.add(sharp_region, blurred_region)
    
    # if 'circle' in visual_prompt_type:
    #     center_coordinates = ((bbox[0]+bbox[2]//2), (bbox[1]+bbox[3]//2))
    #     axes_length = (bbox[2] // 2, bbox[3] // 2)
    #     prompted_image = cv2.ellipse(
    #         prompted_image,
    #         center_coordinates,
    #         axes_length,
    #         0,
    #         0,
    #         360,
    #         color,
    #         thickness,
    #     )
    if 'circle' in visual_prompt_type:
        mask_center, mask_height, mask_width = mask2chw(mask)
        center_coordinates = (mask_center[1], mask_center[0])
        axes_length = (mask_width // 2, mask_height // 2)
        prompted_image = cv2.ellipse(
            prompted_image,
            center_coordinates,
            axes_length,
            0,
            0,
            360,
            color,
            thickness,
        )
    if 'black' in visual_prompt_type:
        prompted_image = cv2.bitwise_and(
            prompted_image.copy(),
            prompted_image.copy(),
            mask=np.clip(mask, 0, 255).astype(np.uint8),
        )
    if visualize:
        cv2.imwrite(os.path.join('masked_img.jpg'), prompted_image)
        prompted_image = Image.fromarray(prompted_image.astype(np.uint8))
    return prompted_image

def gen_gauss_img(mean,sigma,original_img,device):
    # mean均值,sigma标准差,original_img(3*H*W)
    gauss = np.random.normal(mean,sigma,(original_img[1],original_img[2],original_img[0]))
    noisy_img = original_img + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    noisy_img = torch.Tensor(noisy_img)
    return noisy_img

def calc_IoU(res_mask, target, sum_I, sum_U):
    # masks: (N,H,W) 0,1
    # pred: (N)   0-149
    # target: (H,W) 0-149
    target_infos = torch.unique(target).cpu().numpy()
    # 枚举target_info, 取出对应pred对应的mask
    # print(target_infos)
    # print(torch.unique(res_mask).cpu().numpy())
    # u = fk
    for target_info in target_infos:
        gt_now = target == target_info
        pred_now = res_mask == target_info
        I = torch.sum(torch.logical_and(pred_now, gt_now))
        U = torch.sum(torch.logical_or(pred_now, gt_now))
        sum_I[target_info] += I
        sum_U[target_info] += U
    # for i in range(len(masks)):
    #     mask = masks[i]
    #     pred_idx = pred[i]
    #     # 取出target中为pred_idx的部分
    #     target_mask = target == pred_idx
    #     I = torch.sum(torch.logical_and(mask, target_mask))
    #     U = torch.sum(torch.logical_or(mask, target_mask))
        
    #     # if U == 0:
    #     #     this_iou = 0.0
    #     # else:
    #     #     this_iou = I * 1.0 / U
    #     # I, U = I, U

    #     sum_I[pred_idx-1] += I
    #     sum_U[pred_idx-1] += U
    return sum_I, sum_U

def Compute_IoU(pred, target, cum_I, cum_U, mean_IoU=[]):

    if target.dtype != torch.bool:
        target = target.type(torch.bool).squeeze(0)

    I = torch.sum(torch.logical_and(pred, target))
    U = torch.sum(torch.logical_or(pred, target))

    if U == 0:
        this_iou = 0.0
    else:
        this_iou = I * 1.0 / U
    I, U = I, U


    cum_I += I
    cum_U += U
    mean_IoU.append(this_iou)

    return this_iou, mean_IoU, cum_I, cum_U

def AisPartOfB(result_seg,this_seg):
    # result_seg: (H,W)
    # this_seg: (H,W)
    
    I = torch.sum(torch.logical_and(result_seg, this_seg))
    U = torch.sum(torch.logical_or(result_seg, this_seg))
    if (1-(this_seg&result_seg).sum()/result_seg.sum() < 0.1) & (I/U > 0.5):
        return True
    else:
        return False

def show_masks3(anns, anns2, anns3):
    if type(anns) ==  torch.Tensor:
        if len(anns.shape) == 2:
            anns = anns.unsqueeze(0)
        anns = anns.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        anns = anns[None,:,:]
    if type(anns2) ==  torch.Tensor:
        if len(anns2.shape) == 2:
            anns2 = anns2.unsqueeze(0)
        anns2 = anns2.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        anns2 = anns2[None,:,:]
    if type(anns3) ==  torch.Tensor:
        if len(anns3.shape) == 2:
            anns3 = anns3.unsqueeze(0)
        anns3 = anns3.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        anns3 = anns3[None,:,:]
    
    # if type(targets) ==  torch.Tensor: 
    #     targets = targets.bool().cpu().numpy()[0] # Tensor를 cpu로 옮기고 numpy로 변환
     
    if len(anns) == 0:
        return

    h,w = anns.shape[-2:]
    
    ax = plt.gca() # 현재의 axis 가져오기
    ax.set_autoscale_on(False)

    # sorted_anns = sorted(anns, key=lambda x: -np.sum(x)) # mask size 오름차순으로 정렬

    img = np.ones((h, w*4, 4)) # 빈 이미지 생성
    img[:,:,3] = 0 

    for ann,ann2,ann3 in zip(anns,anns2,anns3):
        color_mask = np.concatenate([np.random.random(3), [0.7]]) # 마스크 랜덤 색상 지정 및 투명도 설명
        img[:, w:2*w, :][ann] = color_mask # 빈 이미지에 마스크 그려넣기
        img[:, 2*w:3*w, :][ann2] = color_mask
        img[:, 3*w:4*w, :][ann3] = color_mask
    ax.imshow(img)

def show_masks2(anns, targets):
    if type(anns) ==  torch.Tensor:
        if len(anns.shape) == 2:
            anns = anns.unsqueeze(0)
        anns = anns.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        anns = anns[None,:,:]
    if type(targets) ==  torch.Tensor:
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(0)
        targets = targets.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        targets = targets[None,:,:]
    
    # if type(targets) ==  torch.Tensor: 
    #     targets = targets.bool().cpu().numpy()[0] # Tensor를 cpu로 옮기고 numpy로 변환
     
    if len(anns) == 0:
        return

    h,w = anns.shape[-2:]
    
    ax = plt.gca() # 현재의 axis 가져오기
    ax.set_autoscale_on(False)

    # sorted_anns = sorted(anns, key=lambda x: -np.sum(x)) # mask size 오름차순으로 정렬

    img = np.ones((h, w*2, 4)) # 빈 이미지 생성
    img[:,:,3] = 0 

    for ann,target in zip(anns,targets):
        color_mask = np.concatenate([np.random.random(3), [0.7]]) # 마스크 랜덤 색상 지정 및 투명도 설명
        img[:, :w, :][ann] = color_mask # 빈 이미지에 마스크 그려넣기
        img[:, w:, :][target] = color_mask
    ax.imshow(img)

def show_masks1(anns):
    if type(anns) ==  torch.Tensor:
        if len(anns.shape) == 2:
            anns = anns.unsqueeze(0)
        anns = anns.cpu().numpy() # Tensor를 cpu로 옮기고 numpy로 변환
    else:
        anns = anns[None,:,:]
    
    # if type(targets) ==  torch.Tensor: 
    #     targets = targets.bool().cpu().numpy()[0] # Tensor를 cpu로 옮기고 numpy로 변환
     
    if len(anns) == 0:
        return

    h,w = anns.shape[-2:]
    
    ax = plt.gca() # 현재의 axis 가져오기
    ax.set_autoscale_on(False)

    # sorted_anns = sorted(anns, key=lambda x: -np.sum(x)) # mask size 오름차순으로 정렬

    img = np.ones((h, w*2, 4)) # 빈 이미지 생성
    img[:,:,3] = 0 

    for ann in anns:
        color_mask = np.concatenate([np.random.random(3), [0.7]]) # 마스크 랜덤 색상 지정 및 투명도 설명
        img[:, w:2*w, :][ann] = color_mask # 빈 이미지에 마스크 그려넣기
    ax.imshow(img)

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="configs/freesolo/freesolo_30k.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_false", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # for Ref dataset
    parser.add_argument('--clip_model', default='RN50', help='CLIP model name', choices=['RN50', 'RN101', 'RN50x4', 'RN50x64',
                                                                                          'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT0L/14@336px'])
    parser.add_argument('--visual_proj_path', default='./pretrain/', help='')
    parser.add_argument('--dataset', default='refcocog', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--split', default='val', help='only used when testing, testA, testB')
    parser.add_argument('--fusion_mode', default='G2L', help='attn_masking, toekn_masking, L2G, G2L(default), G2L&L2G')
    parser.add_argument('--splitBy', default='umd', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--show_results', action='store_true', help='Whether to show results ')

    return parser


'''
For multi_head_attention In CLIP's text encoder
'''

class MHA(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        self.attention_map = None
        self.attention_map_gradients = None

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MHA, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


    def multi_head_attention_forward(self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                           value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            static_k, static_v: static key and value used for attention operators.


        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
              will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        self.tgt_len, self.bsz, self.embed_dim = tgt_len, bsz, embed_dim

        src_len, _, _ = key.shape
        self.src_len = src_len

        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], \
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def _scaled_dot_product_attention(self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.

        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.

        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.

            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """


        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        self.save_attn_map(attn) # modified
        if attn_mask is not None:
            attn += attn_mask
        attn = softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)

        gradient_hook = attn.register_hook(self.save_attn_gradient) # modified
        # print(attn_gradients)

        # print(attn_gradients, 'attn_gradients')
        # print(attn.grad, ' attn.grad')
        # print(attn_gradients, 'attn_gradients')

        return output, attn

    def save_attn_gradient(self, gradient): # modified
        gradient = gradient.view(self.bsz, self.num_heads, self.tgt_len, self.src_len)
        gradient = gradient.sum(dim=1) / self.num_heads
        self.attention_map_gradients = gradient

    def save_attn_map(self, attn):
        attn = attn.view(self.bsz, self.num_heads, self.tgt_len, self.src_len)
        attn = attn.sum(dim=1) / self.num_heads
        self.attention_map = attn

    def get_attn_map(self):
        return self.attention_map

    def get_attn_gradients(self):
        return self.attention_map_gradients


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)
