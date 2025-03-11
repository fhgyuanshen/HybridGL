import argparse
import pickle
import torch
print(torch.__version__)
import os
Height, Width = 224, 224
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy
import numpy as np
import tqdm
import copy


# hacky way to register
# import freesolo.data.datasets.builtin
# from freesolo.modeling.solov2 import PseudoSOLOv2

# refer dataset
from data.dataset_refer_bert_gem import ReferDataset4
from utils import default_argument_parser,extract_noun_phrase

# export CUDA_VISIBLE_DEVICES=3

from typing import List
import cv2
import gem

# export CUDA_VISIBLE_DEVICES=3

def main(args, Height, Width, Neg):
    assert args.eval_only, 'Only eval_only available!'

    if args.dataset == 'refcocog':
        args.splitBy = 'umd'  # umd or google in refcocog
    else:
        args.splitBy = 'unc'  # unc in refcoco, refcoco+,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("now using cuda: ",torch.version.cuda)
        print("gpu cnt:",torch.cuda.device_count())
        # for gpucnt in range(torch.cuda.device_count()):
        #     print(gpucnt,' : ',torch.cuda.get_device_name(gpucnt))
    else :
        print("now using CPU")
    model_name = "ViT-B/16"
    pretrained = "openai"
    # model_name = 'ViT-B-16-quickgelu'
    # pretrained = 'metaclip_400m'
    gem_model = gem.create_gem_model(
        model_name=model_name, pretrained=pretrained, device=device
    )
    preprocess = gem.get_gem_img_transform()
    dataset = ReferDataset4(args,
                           image_transforms=None,
                           target_transforms=None,
                           split=args.split,
                           eval_mode=True,
                           preprocesser=preprocess)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    
    nlp = spacy.load('en_core_web_lg')
    mode = 'ViT'  # or ViT
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'

    tbar = tqdm.tqdm(data_loader)
    # maskss=[]
    # boxess=[]
    for i, data in enumerate(tbar):
        # if i < 356 or i>4181:
        #     continue
        # if i < 4178 and i> 358:
        #     continue
        with torch.no_grad():
            image, target, sentence_raw = data
            
            # ############### A D D #############################
            tensor_img = (image[0]['tensor_img']).to(device)
            W = int(image[0]['width'])
            H = int(image[0]['height'])
            # print("tensor_img:", tensor_img.shape)
            j=0
            for sentence, j in zip(sentence_raw, range(len(sentence_raw))):
                sentence = sentence[0].lower()
                doc = nlp(sentence)
                # print(i, j, "sentence:", sentence)
                sentence_for_spacy = []

                for cntdoc, token in enumerate(doc):
                    if token.text == ' ':
                        continue
                    sentence_for_spacy.append(token.text)

                sentence_for_spacy = ' '.join(sentence_for_spacy)
                noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
                # imgattn = genattn_withdiff(diff_img, ldm_stable, prompt = sentence_for_spacy, target_noun = head_noun, time_step=NUM_DIFFUSION_STEPS, generator=g_cpu, controller =controller)
                imgattn = gem_model(tensor_img, [noun_phrase])[0][0]
                imgattn = imgattn.cpu()
                imgattn = torch.Tensor(cv2.resize(imgattn.numpy(), (W, H), interpolation=cv2.INTER_LINEAR))
                # print("imgattn:", imgattn.shape)
                # if i==0 and j==0:
                #     print("head_noun:", head_noun)
                #     heatmap = imgattn.cpu().numpy()
                #     print(heatmap[0][0])
                #     # heatmap = cv2.resize(heatmap, (W, H))
                #     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                #     cv2.imwrite(f'heatmap{i}.jpg', heatmap)
                    
            
                path2attn=f"./data/var/{args.dataset}/attns_gem_{args.split}_openai_b16/"
                if not os.path.exists(path2attn):
                    os.makedirs(path2attn)
                    print("Folder created")
                fattn = open(path2attn+"attnvar"+str(i)+"j"+str(j),'wb')
                # pickle.dump(maskss, fmask)
                pickle.dump(imgattn, fattn)
                fattn.close()
        
    
    # fmask = open("./data/cogmaskvar",'wb')
    # fmask.close()
    print("save attns finish!!!")




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl', 'MODEL.WEIGHTS', 'checkpoints/FreeSOLO_R101_30k_pl.pth']
    # opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl', 'MODEL.WEIGHTS', 'checkpoints/FreeSOLO_R101_30k_pl.pth']
    args.opts = opts
    # print(args.opts)
    Negs=[10000]
    for Neg in Negs:
        print("Neg:",Neg)
        main(args, Height, Width, Neg)
        print("Save Vars Finished!!!")

