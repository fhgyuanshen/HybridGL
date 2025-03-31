import torch
import torch.nn as nn
import clip
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import copy


class CLIPViTFM(nn.Module):  
    def __init__(self, model_name='ViT-B/16', size=224):
        super().__init__()

        if model_name == 'ViT-B/32':
            self.last_layer = 10
            self.num_heads = 12
        elif model_name == 'ViT-B/16':
            self.last_layer = 10
            self.num_heads = 12

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
        N, H, W = pred_masks.size()
        attn_masks = torch.ones((N * self.num_heads, H*W+1, H*W+1), dtype=torch.bool).to(self.device)
        attn_masks[:, 0, 1:] = pred_masks[:,None,:,:].expand(-1,self.num_heads,-1,-1).reshape(N * self.num_heads, -1)
        return ~attn_masks
    
    def forward(self, local_imgs, global_imgs, pred_masks, masking_block=None, fusion_mode='G2L'):
        if masking_block is None:
            masking_block = self.last_layer

        vit = self.model.visual

        x = local_imgs.type(self.model.dtype)
        

        if fusion_mode == 'crop': # [1, 512]
            x = vit(x)
            return x[:, 0, :]

        x = vit.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ####################### x2 ####################################
        if global_imgs is not None:
            x2 = global_imgs.type(self.model.dtype)
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
        if fusion_mode == 'token_masking':
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

        elif fusion_mode == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = resblock(x, attn_mask=attn_mask)

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = resblock(x)

        elif fusion_mode == 'L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    x_ori_local = x.clone()
                    x = resblock(x)
                    x2 = resblock(x_ori_local+x2*2,attn_mask=attn_mask) 
                else:                    
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x2 = x2.permute(1, 0, 2) # [46, 50, 768]
                    x2 = self.model.visual.ln_post(x2[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x2 = x2 @ self.model.visual.proj # [46, 512] 
                    return x2

        elif fusion_mode == 'G2L':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    x_ori_global = x2.clone()
                    cls = x_ori_global[:1,:,:]
                    x_ori_global = x_ori_global[1:,:,:]
                    x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x_ori_global.size(0)
                    x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                    x = resblock(x_ori_global*2+x) 
                    x2 = resblock(x2, attn_mask=attn_mask)
                else:
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x = x.permute(1, 0, 2) # [46, 50, 768]
                    x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x = x @ self.model.visual.proj # [46, 512] 

                    return x

        elif fusion_mode == 'G2L&L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            # local
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 

                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                        x_hybrid_local = x.clone()
                        x_hybrid_global = x2.clone()

                    x_ori_local = x.clone()
                    x_ori_global = x2.clone()
                    # token masking
                    cls = x_ori_global[:1,:,:]
                    x_ori_global = x_ori_global[1:,:,:]
                    x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]
                    x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                    x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x_ori_global.size(0)
                    x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                    x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                    x = resblock(x)
                    x2 = resblock(x2,attn_mask=attn_mask)
                    x_hybrid_local = resblock(x_hybrid_local+2*x_ori_global) 
                    x_hybrid_global = resblock(x_ori_local+2*x_hybrid_global, attn_mask=attn_mask)
                    
                else:
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x_hybrid_local = x_hybrid_local.permute(1, 0, 2) # [46, 50, 768]
                    x_hybrid_local = self.model.visual.ln_post(x_hybrid_local[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x_hybrid_local = x_hybrid_local @ self.model.visual.proj # [46, 512] 

                    x_hybrid_global = x_hybrid_global.permute(1, 0, 2) # [46, 50, 768]
                    x_hybrid_global = self.model.visual.ln_post(x_hybrid_global[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x_hybrid_global = x_hybrid_global @ self.model.visual.proj # [46, 512] 
                    return x_hybrid_local + x_hybrid_global


        return x

