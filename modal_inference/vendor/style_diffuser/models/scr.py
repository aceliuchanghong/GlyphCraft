from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_backbone import FrozenVisionBackbone


def paired_info_nce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    原生 PyTorch 数值稳定 Paired InfoNCE（FP32 计算，零第三方依赖）：
    query:     [B, D] (L2 normalized)
    positive:  [B, D] (L2 normalized)
    negatives: [B, K, D] (L2 normalized)
    """
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negatives.float(), dim=-1)

    pos_logit = (q * p).sum(dim=-1, keepdim=True) / temperature  # [B, 1]
    neg_logits = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
    logits = torch.cat([pos_logit, neg_logits], dim=-1)  # [B, 1 + K]
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


class _LayerProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SCRv2(nn.Module):
    """
    SCR 2.0（现代轻量多尺度风格对比模块）：
    - 彻底淘汰 71M 从零训 VGG-BN，复用冻结 DINOv2 多层 [CLS | mean | std] 统计表征；
    - 可训练投影头仅 ~3M 参数，全程 LayerNorm（无 BatchNorm2d，兼容任意 batch_size>=1）；
    - 配合「同风格 × 异内容」正样本与「同内容 × 异风格」负样本，5k-10k 步即可高纯度解耦风格。
    """

    def __init__(
        self,
        vision_backbone: FrozenVisionBackbone,
        feature_layers: Sequence[int] = (3, 6, 9, 12),
        proj_dim: int = 256,
        temperature: float = 0.07,
        mode: str = "training",
        neg_chunk: int = 8,
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.feature_layers = tuple(int(i) for i in feature_layers)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.neg_chunk = max(1, int(neg_chunk))

        in_dim = int(vision_backbone.hidden_size) * 3  # [CLS, mean_patch, std_patch]
        hidden_dim = max(512, proj_dim * 2)
        self.projectors = nn.ModuleList(
            [
                _LayerProjector(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
                for _ in self.feature_layers
            ]
        )

        # 始终冻结底层视觉基座；仅按 mode 控制投影头是否可训练
        self.vision_backbone.requires_grad_(False)
        trainable = mode == "training"
        self.projectors.requires_grad_(trainable)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """仅保存轻量投影头权重（~12 MB），不包含冻结 DINOv2。"""
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("vision_backbone.")
        }

    def load_trainable_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        cleaned = {
            k: v for k, v in state_dict.items() if not k.startswith("vision_backbone.")
        }
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        missing_proj = [k for k in missing if not k.startswith("vision_backbone.")]
        if strict and (missing_proj or unexpected):
            raise RuntimeError(
                f"加载 SCR 2.0 权重失败: missing={missing_proj[:6]}, unexpected={unexpected[:6]}"
            )

    def encode_raw_features(self, raw_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """直接将预缓存的多层 DINOv2 统计向量投影至超球面（零 DINOv2 前向开销）。"""
        proj_dtype = self.projectors[0].net[1].weight.dtype
        return [
            proj(feat.to(dtype=proj_dtype))
            for proj, feat in zip(self.projectors, raw_feats)
        ]

    def encode_style_layers(self, images: torch.Tensor) -> List[torch.Tensor]:
        raw_feats = self.vision_backbone.extract_multilayer_features(
            images, layers=self.feature_layers
        )
        return self.encode_raw_features(raw_feats)

    def _encode_negatives(self, neg_images: torch.Tensor) -> torch.Tensor:
        """neg_images: [B, K, 3, H, W] -> [num_layers, B, K, proj_dim]"""
        b, num_neg = neg_images.shape[:2]
        chunk = min(self.neg_chunk, num_neg)
        parts: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, num_neg, chunk):
                piece = neg_images[:, start : start + chunk]
                count = piece.shape[1]
                flat = piece.reshape(b * count, *piece.shape[2:])
                layer_embs = self.encode_style_layers(flat)
                stacked = torch.stack(
                    [emb.view(b, count, -1) for emb in layer_embs], dim=0
                )
                parts.append(stacked)
        return torch.cat(parts, dim=2)

    def forward(
        self,
        sample_imgs: Optional[torch.Tensor],
        pos_imgs: Optional[torch.Tensor],
        neg_imgs: Optional[torch.Tensor],
        sample_raw_feats: Optional[Sequence[torch.Tensor]] = None,
        pos_raw_feats: Optional[Sequence[torch.Tensor]] = None,
        neg_raw_feats: Optional[Sequence[torch.Tensor]] = None,
        num_neg: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        if sample_raw_feats is not None:
            sample_embs = self.encode_raw_features(sample_raw_feats)
        else:
            assert sample_imgs is not None
            sample_embs = self.encode_style_layers(sample_imgs)

        bsz = sample_embs[0].shape[0]
        proj_trainable = self.projectors[0].net[1].weight.requires_grad

        if proj_trainable:
            if pos_raw_feats is not None:
                pos_embs = self.encode_raw_features(pos_raw_feats)
            else:
                assert pos_imgs is not None
                pos_embs = self.encode_style_layers(pos_imgs)

            if neg_raw_feats is not None:
                assert num_neg is not None
                layer_embs = self.encode_raw_features(neg_raw_feats)
                neg_embs = torch.stack(
                    [emb.view(bsz, int(num_neg), -1) for emb in layer_embs], dim=0
                )
            else:
                assert neg_imgs is not None
                b, k_neg = neg_imgs.shape[:2]
                chunk = min(self.neg_chunk, k_neg)
                parts: List[torch.Tensor] = []
                for start in range(0, k_neg, chunk):
                    piece = neg_imgs[:, start : start + chunk]
                    count = piece.shape[1]
                    flat = piece.reshape(b * count, *piece.shape[2:])
                    layer_embs = self.encode_style_layers(flat)
                    parts.append(
                        torch.stack([emb.view(b, count, -1) for emb in layer_embs], dim=0)
                    )
                neg_embs = torch.cat(parts, dim=2)
        else:
            with torch.no_grad():
                if pos_raw_feats is not None:
                    pos_embs = self.encode_raw_features(pos_raw_feats)
                else:
                    assert pos_imgs is not None
                    pos_embs = self.encode_style_layers(pos_imgs)

                if neg_raw_feats is not None:
                    assert num_neg is not None
                    layer_embs = self.encode_raw_features(neg_raw_feats)
                    neg_embs = torch.stack(
                        [emb.view(bsz, int(num_neg), -1) for emb in layer_embs], dim=0
                    )
                else:
                    assert neg_imgs is not None
                    neg_embs = self._encode_negatives(neg_imgs)
        return sample_embs, pos_embs, neg_embs

    def calculate_nce_loss(
        self,
        sample_s: Sequence[torch.Tensor],
        pos_s: Sequence[torch.Tensor],
        neg_s: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        计算多层 InfoNCE 风格对比损失；支持传入按时间步 SNR 门控的逐样本权重 sample_weights [B]。
        """
        num_layers = len(sample_s)
        total = torch.tensor(0.0, device=sample_s[0].device, dtype=torch.float32)
        for idx, (q, p) in enumerate(zip(sample_s, pos_s)):
            q_n = F.normalize(q.float(), dim=-1)
            p_n = F.normalize(p.float(), dim=-1)
            n_n = F.normalize(neg_s[idx].float(), dim=-1)
            pos_logit = (q_n * p_n).sum(dim=-1, keepdim=True) / self.temperature
            neg_logits = torch.bmm(n_n, q_n.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_logit, neg_logits], dim=-1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            per_sample = F.cross_entropy(logits, labels, reduction="none")
            if sample_weights is not None:
                w = sample_weights.float().to(per_sample.device)
                layer_loss = (per_sample * w).sum() / w.sum().clamp_min(1e-6)
            else:
                layer_loss = per_sample.mean()
            total = total + layer_loss
        return total / max(num_layers, 1)


# 别名保持导入简洁
SCR = SCRv2
