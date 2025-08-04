import torch
import numpy as np
import timm
import Model.CascadeHead as ch 


class MSA(torch.nn.Module):
    def __init__(self, backbone, backbone_out, layers=4, layer_dims=[2, 50, 404, 555], single=False):
        super(MSA, self).__init__()
        self.backbone = backbone
        self.global_pool = backbone.head.global_pool
        self.backbone.head = torch.nn.Identity()
        self.pixel_classifier = torch.nn.Linear(in_features=1024, out_features=48*48*3, bias=True)
        self.softmax = torch.nn.Softmax(dim=0)
        self.upsample = torch.nn.Upsample(scale_factor=8, mode='nearest')
        self.mask_attention = torch.nn.MultiheadAttention(embed_dim=backbone_out, num_heads=8)
        self.mask_feed_forward = torch.nn.Sequential(
                                torch.nn.Linear(backbone_out, 1024),
                                torch.nn.LayerNorm(1024)
                            )

        if single:
            self.classifier = torch.nn.Linear(in_features=1024, out_features=200, bias=True)
        else:
            self.classifier = ch.CascadeHead(backbone_out=1536, layers=layers, layer_dims=layer_dims, activation=torch.nn.ReLU, cumulative=True)
        
        #self.fusion_attention = torch.nn.MultiheadAttention(embed_dim=backbone_out, num_heads=4)
        #self.feed_forward = torch.nn.Sequential(
        #                        torch.nn.Linear(backbone_out, 512),
        #                        torch.nn.LayerNorm(512)
        #                    )


    def forward(self, x0):
        x = self.backbone.forward_intermediates(x0, indices=[0,1,2,3])[1]
       
        x[0] = self.global_pool(x[0].swapaxes(1,2).swapaxes(2,3))
        x[1] = self.global_pool(x[1].swapaxes(1,2).swapaxes(2,3))
        x[2] = self.global_pool(x[2].swapaxes(1,2).swapaxes(2,3))
        x[3] = self.global_pool(x[3].swapaxes(1,2).swapaxes(2,3))

        # Masking Module
        x = torch.concat((x[0],x[1],x[2],x[3]), dim=1)
        x = self.mask_attention(x, x, x)[0]
        x = self.mask_feed_forward(x)

        x = self.pixel_classifier(x)
        x = x0 * self.upsample(self.softmax(x).reshape(3,48,48).unsqueeze(0)).squeeze()
        
        # Done
        x = self.backbone.forward(x)
        x = self.global_pool(x)

        # Do stuff with features
        x = self.classifier(x)

        return x


