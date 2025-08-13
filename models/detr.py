import torchvision.models as models
import torch.nn as nn
import torch
import math


class PositionEncodingSineCosine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, mask):
        # mask: [B, H, W] – True for padded pixels
        not_mask = ~mask  # flip it: 1s where valid
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(
            0, 3, 1, 2
        )  # Returns [B, 2*num_pos_feats, H, W]
        return pos


class BackboneWithPE(nn.Module):
    def __init__(self, backbone, position_encoding):
        super().__init__()
        self.body = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # remove FC and AvgPool
        # For ResNet-50, this leaves output of shape [B, 2048, H/32, W/32].
        self.num_channels = 2048
        self.position_encoding = position_encoding

    def forward(self, x):
        features = self.body(x)  # [B, 2048, H, W]
        mask = torch.zeros(
            (x.shape[0], features.shape[-2], features.shape[-1]),
            dtype=torch.bool,
            device=x.device,
        )
        pos = self.position_encoding(
            mask
        )  # [B, d_model, H, W] / d_model = 2 * num_pos_feats
        return features, mask, pos


class SimpleTransformer(nn.Module):
    def __init__(
        self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_decoder_layers
        )
        self.d_model = d_model

    def forward(self, src, mask, query_embed, pos_embed):
        # src: [HW, B, C]
        memory = self.encoder(src + pos_embed, src_key_padding_mask=mask)
        tgt = torch.zeros_like(query_embed)  # [num_queries, B, d_model]
        hs = self.decoder(
            tgt + query_embed,
            memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mask,
        )
        return hs  # [num_queries, B, d_model]


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.backbone = backbone  # e.g., ResNet
        self.transformer = transformer  # Encoder-Decoder transformer
        self.num_queries = num_queries

        # Learnable object queries: [num_queries, d_model]
        self.query_embed = nn.Embedding(num_queries, transformer.d_model)

        self.input_proj = nn.Conv2d(
            backbone.num_channels, transformer.d_model, kernel_size=1
        )

        # Prediction heads:
        self.class_embed = nn.Linear(
            transformer.d_model, num_classes + 1
        )  # +1 for "no-object" class
        self.bbox_embed = MLP(
            transformer.d_model, transformer.d_model, output_dim=4, num_layers=3
        )

    def forward(self, samples):
        # samples: tensor image OR NestedTensor (image + mask)
        features, mask, pos = self.backbone(samples)
        src = self.input_proj(features)  # [B, d_model, H, W]

        # Positional encoding
        # [B, d_model, H, W]

        # Flatten inputs for transformer
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        pos = pos.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            1, B, 1
        )  # [num_queries, B, d_model]

        # Transformer forward
        hs = self.transformer(
            src, mask.flatten(1), query_embed, pos
        )  # output: [num_layers, num_queries, B, d_model]

        # print("hs.shape:", hs.shape)

        outputs_class = self.class_embed(
            hs
        )  # [num_queries=100, batch_size=16, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()

        pred_logits = outputs_class.permute(
            1, 0, 2
        )  # [batch_size, num_queries, num_classes+1]
        pred_boxes_cxcywh = outputs_coord.permute(1, 0, 2)  # [batch_size, num_queries, 4]
        # Convert to (x_min, y_min, x_max, y_max)
        x_min = pred_boxes_cxcywh[..., 0]
        y_min = pred_boxes_cxcywh[..., 1]
        w = pred_boxes_cxcywh[..., 2]
        h = pred_boxes_cxcywh[..., 3]

        x_max = x_min + w
        y_max = y_min + h

        # Clamp to [0,1] just to be safe (avoid boxes going outside image)
        x_max = x_max.clamp(0, 1)
        y_max = y_max.clamp(0, 1)

        # Stack back
        pred_boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes_xyxy}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_model(num_classes, num_queries, d_model, device):
    backbone = BackboneWithPE(
        models.resnet50(pretrained=True), PositionEncodingSineCosine(num_pos_feats=128)
    )
    transformer = SimpleTransformer(
        d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6
    )
    model = DETR(
        backbone, transformer, num_classes=num_classes, num_queries=num_queries
    )
    model.to(device)
    return model
