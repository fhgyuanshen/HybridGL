import torch
import torch.nn as nn
import clip
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import copy


class CLIPViTFM2(nn.Module):  
    def __init__(self, model_name='ViT-B/16', size=224):
        super().__init__()

        if model_name == 'ViT-B/32':
            self.last_layer = 10
            self.num_heads = 12
        elif model_name == 'ViT-B/16':
            self.last_layer = 10
            self.num_heads = 12
        # elif model_name == 'ViTB16_quickgelu':
        #     self.last_layer = 10
        #     self.num_heads = 12
        elif model_name == 'ViT-L/14':
            self.last_layer = 23
            self.num_heads = 16

        self.model, _ = clip.load(model_name)


    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def text_masking_feature(self, text, masking_index=[], masking_block=11):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token
 
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x) 
                else:
                    x = resblock(x)
            else:
                x = resblock(x)
 
        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]

        return x
    
    def text_feature(self, text):
        text_encoder = self.model.transformer 
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]
 
        x = text_encoder.resblocks(x)

        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]
        
        return x
    


    def calculate_score(self, image_features, text_features, visual_norm_dim=1):
        # image_features = [N,512]
        # text_feature = [1,512]
        # logit_scale.exp() = 100.

        image_features = image_features / image_features.norm(dim=visual_norm_dim, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
         
        return logits_per_image # [N, 1]

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        n = int(np.sqrt(N))
        assert n * n == N

        emb = emb.permute(1, 0)
        emb = emb.view(1, D, n, n).contiguous()
        emb = F.upsample(emb, size=(self.size), mode='bilinear',
                         align_corners=None)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def make_attn_mask(self, pred_masks, size=None):
        # pred_masks =  [46, H, W], Torch.bool
        if size is not None:
            pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]
        # pred_masks = pred_masks.type(torch.bool)
        N, H, W = pred_masks.size()
        attn_masks = torch.ones((N * self.num_heads, H*W+1, H*W+1), dtype=torch.bool).to(self.device)
        attn_masks[:, 0, 1:] = pred_masks[:,None,:,:].expand(-1,self.num_heads,-1,-1).reshape(N * self.num_heads, -1)
        # attn_masks[:, 1:, 0] = pred_masks[:,None,:,:].repeat(1,self.num_heads,1,1).view(N * self.num_heads, -1)
        return ~attn_masks
    
    def make_attn_mask_little(self, pred_masks, size=None):
        # pred_masks =  [46, H, W], Torch.bool
        if size is not None:
            pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]
        # pred_masks = pred_masks.type(torch.bool)
        N, H, W = pred_masks.size()
        attn_masks = torch.ones((N * self.num_heads, H*W+1, H*W+1), dtype=torch.float32).to(self.device)
        attn_masks[:, 0, 1:] = pred_masks[:,None,:,:].expand(-1,self.num_heads,-1,-1).reshape(N * self.num_heads, -1)
        # attn_masks[:, 1:, 0] = pred_masks[:,None,:,:].repeat(1,self.num_heads,1,1).view(N * self.num_heads, -1)
        # print(attn_masks-1)
        return (attn_masks-1)*4
    
    def forward(self, masked_img, prompted_img, pred_masks,  masking_block=None, masking_type='share_attn_merge_g2l_tokenmask', layer_add=0):
        if masking_block is None:
            masking_block = self.last_layer

        vit = self.model.visual

        x = masked_img.type(self.model.dtype)
        

        if masking_type == 'crop': # [1, 512]
            x = vit(x)
            # print(x[:,0,:].shape)
            return x[:, 0, :]

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]

        # size = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + vit.positional_embedding.to(x.dtype)
        # x = x + self.original_pos_embedding
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ####################### x2 ####################################
        if prompted_img is not None:
            x2 = prompted_img.type(self.model.dtype)
            x2 = vit.conv1(x2)  # shape = [*, width, grid, grid]
            x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
            x2 = x2.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x2 = torch.cat([vit.class_embedding.to(x2.dtype) + torch.zeros(x2.shape[0], 1, x2.shape[-1],
                                                                        dtype=x2.dtype, device=x2.device), x2], dim=1)  # shape = [*, grid ** 2 + 1, width]

            x2 = x2 + vit.positional_embedding.to(x2.dtype)
            x2 = vit.ln_pre(x2)

            x2 = x2.permute(1, 0, 2)  # NLD -> LND



        L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)
        size = int(np.sqrt(L))
        assert size * size == L

        pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))
        # pred_masks = torch.stack([pred_mask.view(size,224//size,size,224//size).max(dim=1)[0].max(dim=2)[0] for pred_mask in pred_masks])

        if masking_type == 'token_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    cls = x[:1,:,:]
                    x = x[1:,:,:]
                    x = x.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1,N,-1), x], dim=0) # [50, 46, 768]
                    x = resblock(x) # [50, 46, 768]

                    if block_idx == self.last_layer+1:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = resblock(x)

        elif masking_type == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = resblock(x, attn_mask=attn_mask)
                    # print(x.shape,block_idx)
                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = resblock(x)

        elif masking_type == 'share_attn_merge_l2g':
            attn_mask = self.make_attn_mask(pred_masks)
            # local
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                # print(block_idx,x.shape)
                if block_idx >= masking_block:
                    # x2 = resblock(x2,attn_weights=attn_weights)
                    _x = x.clone()
                    
                    x = resblock(x)
                    # x2 = resblock((_x+x2)/2, x_k=(_x+x2)/2, x_v=(_x+x2)/2,attn_mask=attn_mask)
                    x2 = resblock(_x+x2*2,attn_mask=attn_mask) 
                    # print(x2.shape,x.shape)
                else:
                    x = resblock(x) 
                    x2 = resblock(x2)
            
                if block_idx == self.last_layer+1:
                    x2 = x2.permute(1, 0, 2) # [46, 50, 768]
                    x2 = self.model.visual.ln_post(x2[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x2 = x2 @ self.model.visual.proj # [46, 512] 
                    return x2
                    # break

        elif masking_type == 'share_attn_merge_g2l_tokenmask':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            # print(x1_x2.shape)
            # local
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                # print(block_idx,x.shape)
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    _x2 = x2.clone()
                    cls = _x2[:1,:,:]
                    _x2 = _x2[1:,:,:]
                    _x2 = _x2.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    _x2 = _x2.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    _x2 = torch.mul(_x2, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = _x2.size(0)
                    _x2 = _x2.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    _x2 = _x2.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    _x2 = torch.cat([cls.expand(-1,N,-1), _x2], dim=0) # [50, 46, 768]

                    x = resblock(_x2*2+x) 
                    x2 = resblock(x2, attn_mask=attn_mask)
                    # print(x2.shape,x.shape)
                else:
                    # x = resblock(x) 
                    # x2 = resblock(x2)
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x = x.permute(1, 0, 2) # [46, 50, 768]
                    x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x = x @ self.model.visual.proj # [46, 512] 

                    return x

        elif masking_type == 'share_attn_merge_gl2gl_new':
            attn_mask = self.make_attn_mask(pred_masks)
            # local
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 

                if block_idx >= masking_block:
                    # clone x and x2
                    _x0 = x.clone()#用来融入global
                    _x20 = x2.clone()#用来token masking融入local
                    if block_idx == masking_block:
                        _x1 = x.clone()#用来跑local
                        _x21 = x2.clone()#用来attn masking跑global
                    # token masking
                    cls = _x20[:1,:,:]
                    _x20 = _x20[1:,:,:]
                    _x20 = _x20.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]
                    _x20 = _x20.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                    _x20 = torch.mul(_x20, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = _x20.size(0)
                    _x20 = _x20.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                    _x20 = _x20.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    _x20 = torch.cat([cls.expand(-1,N,-1), _x20], dim=0) # [50, 46, 768]

                    x = resblock(x)
                    x2 = resblock(x2,attn_mask=attn_mask)
                    _x1 = resblock(_x1+2*_x20) 
                    _x21 = resblock(_x0+2*_x21, attn_mask=attn_mask)
                    # print(x2.shape,x.shape)
                else:
                    x = resblock(x) 
                    x2 = resblock(x2)
            
                if block_idx == self.last_layer+1:
                    _x1 = _x1.permute(1, 0, 2) # [46, 50, 768]
                    _x1 = self.model.visual.ln_post(_x1[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        _x1 = _x1 @ self.model.visual.proj # [46, 512] 

                    _x21 = _x21.permute(1, 0, 2) # [46, 50, 768]
                    _x21 = self.model.visual.ln_post(_x21[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        _x21 = _x21 @ self.model.visual.proj # [46, 512] 
                    return _x1, _x21


        return x

