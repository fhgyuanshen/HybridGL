import torch
import clip

import torchvision.transforms as T
import spacy
import numpy as np
import tqdm
import cv2

# refer dataset
from data.dataset_refer_bert import ReferDataset
from model.backbone import CLIPViTFM
from utils import default_argument_parser, Compute_IoU, extract_noun_phrase, gen_dir_mask, extract_dir_phrase, extract_rela_word, relation_boxes, extract_nouns

import gem
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

Height, Width = 224, 224



def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'

    if args.dataset == 'refcocog':
        args.splitBy = 'umd'  # umd or google in refcocog
    else:
        args.splitBy = 'unc'  # unc in refcoco, refcoco+,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("now using cuda: ",torch.version.cuda)
    else :
        print("now using CPU")
        
    gem_model = gem.create_gem_model(
        model_name='ViT-B/16', pretrained='openai', device=device
    )
    preprocess = gem.get_gem_img_transform()
    dataset = ReferDataset(args,
                           image_transforms=None,
                           target_transforms=None,
                           preprocessor = preprocess,
                           split=args.split)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    Model = CLIPViTFM(model_name='ViT-B/16').to(device)
    Model.eval()

    nlp = spacy.load('en_core_web_lg')
    
    cum_I, cum_U =0, 0
    m_IoU = []
    cum_I_final, cum_U_final = 0, 0
    m_IoU_final = []

    r = 0.5
    alpha = 0.6

    softmax0 = torch.nn.Softmax(0)
    softmax0 = softmax0.to(device)
    k1=3    # topk spatial relationship for the target noun
    k2=6    # topk spatial relationship for other nouns
    fusion_mode = args.fusion_mode
    print(f"fusion mode={fusion_mode}")
    sam = sam_model_registry['default'](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=8,
                                                pred_iou_thresh=0.7,
                                                stability_score_thresh=0.7,
                                                crop_n_layers=0,
                                                crop_n_points_downscale_factor=1,
                                                min_mask_region_area=800,)
    tbar = tqdm.tqdm(data_loader)

    ########################## load data ##########################

    for i, data in enumerate(tbar):
        image, target, sentence_raw = data
        target = target.to(device)
        ####### generate masks #######
        original_img = image['image'].to(device)
        sam_img = np.array((image['sam_img'][0]))
        sam_masks = mask_generator.generate(sam_img)
        masks = [torch.tensor(m['segmentation']) for m in sam_masks]
        masks = torch.stack(masks).to(device)
        
        boxes = [m['bbox'] for m in sam_masks]
        boxes = torch.tensor(boxes).to(device)

        ####### preprocess global and local images #######
        pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(masks.device)
        
        global_imgs = []
        local_imgs = []

        imagesrc = image['sam_img']
        blurred = cv2.GaussianBlur(imagesrc[0].numpy().copy(), (15, 15), 0)
        for pred_box, pred_mask in zip(boxes, masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
    
            global_img = imagesrc[0].numpy()
            if type(pred_mask) != type(imagesrc[0].numpy()):
                mask = pred_mask.cpu().numpy()
            sharp_region = cv2.bitwise_and(
                global_img,
                global_img,
                mask=np.clip(mask, 0, 255).astype(np.uint8),
            )
            inv_mask = 1 - mask
            blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
            global_img = cv2.add(sharp_region, blurred_region)

            global_img = T.ToTensor()(global_img)
            global_img = T.Resize((Height, Width), antialias=None)(global_img)
            global_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(global_img)
            global_imgs.append(global_img)
            
            masked_image = original_img * pred_mask[None,None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
            masked_image = T.Resize((Height, Width), antialias=None)(masked_image.squeeze(0))
            local_imgs.append(masked_image.squeeze(0))

        global_imgs = torch.stack(global_imgs,dim=0).to(device)
        local_imgs = torch.stack(local_imgs, dim=0).to(device)

        ####### calculate hybrid features #######
        hybrid_features = Model(local_imgs=local_imgs, global_imgs=global_imgs, pred_masks=masks, fusion_mode=fusion_mode, masking_block=9)

        ####### calculate text features and scores #######
        for j, sentence in enumerate(sentence_raw):
            sentence = sentence[0].lower()
            doc = nlp(sentence)

            sentence_for_spacy = []

            for doccnt, token in enumerate(doc):
                if token.text == ' ':
                    continue
                sentence_for_spacy.append(token.text)

            sentence_for_spacy = ' '.join(sentence_for_spacy)
            dirflag = extract_dir_phrase(sentence_for_spacy, nlp, False)
            visual_feature = hybrid_features
            
            sentence_token = clip.tokenize(sentence_for_spacy).to(device)
            noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
            
            noun_phrase_token = clip.tokenize(noun_phrase).to(device)
            sentence_features = Model.model.encode_text(sentence_token)
            noun_phrase_features = Model.model.encode_text(noun_phrase_token)

            text_ensemble = r * sentence_features + (1-r) * noun_phrase_features
            score_clip = Model.calculate_score(visual_feature, text_ensemble)

            other_noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
            other_noun_features = torch.zeros(1, 512).to(device)
            cnt_other_nouns = 0
            for other_noun in other_noun_phrases:
                noun_token = clip.tokenize('a photo of '+other_noun).to(device)
                other_noun_features += Model.model.encode_text(noun_token)
                cnt_other_nouns += 1
            if cnt_other_nouns != 0:
                other_noun_features = other_noun_features / cnt_other_nouns
                
            score_clip_Neg = Model.calculate_score(visual_feature, other_noun_features)

            max_index_hybrid = torch.argmax(score_clip)
            result_seg_hybrid = masks[max_index_hybrid]

            _, m_IoU, cum_I, cum_U = Compute_IoU(result_seg_hybrid, target, cum_I, cum_U, m_IoU)

            score_clip = softmax0(score_clip)
            score_clip_Neg = softmax0(score_clip_Neg)

            ######## Spatial Relationship Guidance ########
            relaflag = extract_rela_word(sentence_for_spacy, nlp)
            if k1 > len(score_clip):
                k1 = len(score_clip)
            if k2 > len(score_clip_Neg):
                k2 = len(score_clip_Neg)
            _, maxidxs = torch.topk(score_clip.view(-1),k=k1)
            _, maxNegidxs = torch.topk(score_clip_Neg.view(-1),k=k2)

            topscores = np.zeros(k1)
            if len(nouns)==0:   # sentence only has target noun
                for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                    for idx_j in maxidxs:           
                        topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip[idx_j][0],relaflag)
            else:               # sentence has other nouns
                for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                    for idx_j in maxNegidxs:            #idx_j in boxes' idx
                        topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip_Neg[idx_j][0],relaflag)
            
            topscores = torch.Tensor(topscores).to(device)
            topscores = softmax0(topscores)

            ########  Spatial Coherence Guidance ########
            score_gem_list=[]
            imgattn = gem_model(image['tensor_img'].to(device), [noun_phrase])[0]
            imgattn = T.Resize((int(image['height']), int(image['width'])), antialias=True)(imgattn)[0]
            imgattn = imgattn.to(device)

            imgattn = (imgattn-imgattn.min()) / (imgattn.max()-imgattn.min())

            pmask = gen_dir_mask(dirflag, imgattn.shape[0], imgattn.shape[1], imgattn.device)
            imgattn = imgattn * pmask    # Spatial Position Guidance

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

            for idx_i in range(k1):      
                topscores[idx_i]=topscores[idx_i] * (1 - alpha) + alpha * score_gem[maxidxs[idx_i]][0]
            max_index_final = maxidxs[torch.argmax(topscores)]
            result_seg_final = masks[max_index_final]

            _, m_IoU_final, cum_I_final, cum_U_final = Compute_IoU(result_seg_final, target, cum_I_final, cum_U_final, m_IoU_final)


    f = open(f'./result_log/result_log_{args.dataset}_{args.split}.txt', 'a')
    f.write(f'\n\n fusion_mode={fusion_mode} '
            f'\nDataset: {args.dataset} / {args.split} / {args.splitBy}'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\npure hybridgl: {overall:.2f} / {mean_IoU:.2f}')  
    overall_final = cum_I_final * 100.0 / cum_U_final
    mean_IoU_final = torch.mean(torch.tensor(m_IoU_final)) * 100.0

    f.write(f'\nhybridgl w/ spatial guidance: {overall_final:.2f} / {mean_IoU_final:.2f}')
    f.close()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()    
    with torch.no_grad():
        main(args, Height, Width)

