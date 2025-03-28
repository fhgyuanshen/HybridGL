import argparse
import pickle
import torch
print(torch.__version__)
import clip
import os
Height, Width = 224, 224
Neg = 0.25
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer
import tqdm
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from data.dataset_phrasecut import PhraseCutDataset
from model.backbone2 import CLIPViTFM2
from utils import default_argument_parser, setup, Compute_IoU, extract_noun_phrase, gen_dir_mask, extract_dir_phrase, extract_rela_word, relation_boxes, extract_nouns
from collections import defaultdict
import matplotlib.pyplot as plt

# export CUDA_VISIBLE_DEVICES=3 


def main(args, Height, Width, Fun):
    assert args.eval_only, 'Only eval_only available!'

    if args.dataset == 'refcocog':
        args.splitBy = 'umd'  # umd or google in refcocog
    else:
        args.splitBy = 'unc'  # unc in refcoco, refcoco+,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("now using cuda: ",torch.version.cuda)
        print("gpu cnt:",torch.cuda.device_count())
    else :
        print("now using CPU")
    dataset = PhraseCutDataset(split = 'test')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    ################################### load data

    mode = 'ViT'  # or ViT
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'
    
    Model = CLIPViTFM2(model_name='ViT-B/16').to(device)
    Model.eval()

    nlp = spacy.load('en_core_web_lg')
    
    cum_I, cum_U =0, 0
    m_IoU = []
    cum_I_final, cum_U_final = 0, 0
    m_IoU_final = []

    r = 0.5
    softmax0 = torch.nn.Softmax(0)
    softmax0 = softmax0.to(device)
    k1=3
    k2=6

    print(f"Fun={Fun}")
    sam = sam_model_registry['default'](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=64,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100,)
    tbar = tqdm.tqdm(data_loader)

    ########################## load mode

    for i, data in enumerate(tbar):
        image, gt_masks, sentence_raw = (data[0]['image'].to(device), data[0]['gt_masks'], data[0]['phrase'])
        
        original_imgs = torch.stack([T.Resize((height, width))(img.to(device)) for img, height, width in
                                    zip(image, data[0]['height'], data[0]['width'])], dim=0)  # [1, 3, 428, 640] 
        
        sam_img = np.array((data[0]['sam_img'][0]))
        sam_masks = mask_generator.generate(sam_img)
        masks = [torch.tensor(m['segmentation']) for m in sam_masks]
        masks = torch.stack(masks)#.to(device)

        boxes = [m['bbox'] for m in sam_masks]
        boxes = torch.tensor(boxes)
        boxes = boxes.to(device)

        pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(masks.device)
        
        prompt_imgs = []
        imagesrc = data[0]['sam_img']
        blurred = cv2.GaussianBlur(imagesrc[0].numpy().copy(), (15, 15), 0)
        for pred_box, pred_mask in zip(boxes, masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
    
            prompted_image = imagesrc[0].numpy().copy()
            if type(pred_mask) != type(imagesrc[0].numpy()):
                mask = pred_mask.cpu().numpy()
            sharp_region = cv2.bitwise_and(
                prompted_image.copy(),
                prompted_image.copy(),
                mask=np.clip(mask, 0, 255).astype(np.uint8),
            )
            inv_mask = 1 - mask
            blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
            prompted_image = cv2.add(sharp_region, blurred_region)
            prompt_img = T.ToTensor()(prompted_image)
            prompt_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(prompt_img)
            prompt_img = TF.resize(prompt_img.squeeze(0), (Height, Width))
            prompt_imgs.append(prompt_img)

        prompt_imgs = torch.stack(prompt_imgs,dim=0).to(device)
        
        cropped_imgs = []
        for pred_box, pred_mask in zip(boxes, masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            masked_image = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
            masked_image = TF.resize(masked_image.squeeze(0), (Height, Width))
            cropped_imgs.append(masked_image.squeeze(0))

        cropped_imgs = torch.stack(cropped_imgs, dim=0)
        prompt_features = Model(masked_img=cropped_imgs, prompted_img=prompt_imgs, pred_masks=masks, masking_type=Fun, masking_block=9,layer_add=1)

        for sentence, j in zip(sentence_raw, range(len(sentence_raw))):
            sentence = sentence[0].lower()
            target = gt_masks[j].to(device)
            doc = nlp(sentence)

            sentence_for_spacy = []

            for doccnt, token in enumerate(doc):
                if token.text == ' ':
                    continue
                sentence_for_spacy.append(token.text)

            sentence_for_spacy = ' '.join(sentence_for_spacy)
            dirflag = extract_dir_phrase(sentence_for_spacy, nlp, False)
            visual_feature = prompt_features
            
            sentence_token = clip.tokenize(sentence_for_spacy).to(device)
            noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
            noun_phrase_token = clip.tokenize(noun_phrase).to(device)
            sentence_features = Model.get_text_feature(sentence_token) if mode == 'Res' else Model.model.encode_text(sentence_token)
            noun_phrase_features = Model.get_text_feature(noun_phrase_token) if mode == 'Res' else Model.model.encode_text(noun_phrase_token)

            text_ensemble = r * sentence_features + (1-r) * noun_phrase_features
            score_attn = Model.calculate_score(visual_feature, text_ensemble)
            
            noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
            other_noun_features = torch.zeros(1, 512).to(device)
            cnt_other_nouns = 0
            for other_noun in noun_phrases:
                noun_token = clip.tokenize('a photo of '+other_noun).to(device)
                other_noun_features += Model.model.encode_text(noun_token)
                cnt_other_nouns += 1
            if cnt_other_nouns != 0:
                other_noun_features = other_noun_features / cnt_other_nouns
                
            score_clip_Neg = Model.calculate_score(visual_feature, other_noun_features)

            max_index_attn = torch.argmax(score_attn)
            result_seg_attn = masks[max_index_attn]

            _, m_IoU, cum_I, cum_U = Compute_IoU(result_seg_attn, target, cum_I, cum_U, m_IoU)
            
            score_gem_list=[]
            
            relaflag = extract_rela_word(sentence_for_spacy, nlp)
            fattn = open(f"./data/var/PhraseCut/attns_gem_test_openai_b16/attnvar"+str(i)+"j"+str(j),"rb")
            imgattn = pickle.load(fattn)
            fattn.close()
            imgattn = imgattn.to(device)

            imgattn = (imgattn-imgattn.min()) / (imgattn.max()-imgattn.min())

            pmask = gen_dir_mask(dirflag, imgattn.shape[0], imgattn.shape[1], imgattn.device)
            imgattn = imgattn * pmask

            imgattn = imgattn / imgattn.mean()
            
            
            if relaflag == "big":
                black = 1.95
            elif relaflag == "small":
                black = 1.5
            else:
                black = 1.8
            for pred_mask in masks:
                pred_mask = pred_mask.type(torch.uint8)
                score_gemtmp = (imgattn * (2-black) * pred_mask/(pred_mask.sum())).sum() - (imgattn * black * (1 - pred_mask) / ((1 - pred_mask).sum())).sum()
                score_gem_list.append(torch.Tensor([score_gemtmp]))
            score_gem = torch.stack(score_gem_list,dim=0)
            score_gem=score_gem.to(device)
            score_clip = softmax0(score_attn)
            score_clip_Neg = softmax0(score_clip_Neg)
            
            if k1 > len(score_clip):
                k1 = len(score_clip)
            if k2 > len(score_clip_Neg):
                k2 = len(score_clip_Neg)
            _, maxidxs = torch.topk(score_clip.view(-1),k=k1)
            _, maxNegidxs = torch.topk(score_clip_Neg.view(-1),k=k2)

            noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
            topscores = np.zeros(k1)
            if len(nouns)==0:
                for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                    for idx_j in maxidxs:           
                        topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip[idx_j][0],relaflag)
            else:
                for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                    for idx_j in maxNegidxs:            #idx_j in boxes' idx
                        topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip_Neg[idx_j][0],relaflag)
            
            topscores = torch.Tensor(topscores).to(device)
            topscores = softmax0(topscores)
            alpha = 0.6
            for idx_i in range(k1):      
                topscores[idx_i]=topscores[idx_i] * (1 - alpha) + alpha * score_gem[maxidxs[idx_i]][0]
            max_index_final = maxidxs[torch.argmax(topscores)]
            result_seg_final = masks[max_index_final]

            _, m_IoU_final, cum_I_final, cum_U_final = Compute_IoU(result_seg_final, target, cum_I_final, cum_U_final, m_IoU_final)

    f = open('./result_log/result_log_PhraseCut.txt', 'a')
    f.write(f'\n\n CLIP Model: {mode}  Fun={Fun} '
            f'\nDataset: PhraseCut / test'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\n{overall:.2f} / {mean_IoU:.2f}')  
    overall_final = cum_I_final * 100.0 / cum_U_final
    mean_IoU_final = torch.mean(torch.tensor(m_IoU_final)) * 100.0

    f.write(f'\n{overall_final:.2f} / {mean_IoU_final:.2f}')
    f.close()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    Funs=['share_attn_merge_g2l_tokenmask']
    for Fun in Funs:    
        with torch.no_grad():
            main(args, Height, Width, Fun)

