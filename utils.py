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


def relation_boxes(boxi, boxj, scorei, scorej, relaword):
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
    output_seg[:, :, 0] = binary_image 
    output_seg[:, :, 1] = binary_image 
    output_seg[:, :, 2] = binary_image
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
    gauss = np.random.normal(mean,sigma,(original_img[1],original_img[2],original_img[0]))
    noisy_img = original_img + gauss
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    noisy_img = torch.Tensor(noisy_img)
    return noisy_img

def calc_IoU(res_mask, target, sum_I, sum_U):
    target_infos = torch.unique(target).cpu().numpy()
    for target_info in target_infos:
        gt_now = target == target_info
        pred_now = res_mask == target_info
        I = torch.sum(torch.logical_and(pred_now, gt_now))
        U = torch.sum(torch.logical_or(pred_now, gt_now))
        sum_I[target_info] += I
        sum_U[target_info] += U
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
