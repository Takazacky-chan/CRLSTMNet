# models/crlstmnet.py - 核心模型实现 (修正版)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRLSTMNetLite(nn.Module):
    """
    融合CRNet空间特征提取和LSTM时序建模的轻量版本
    参考CsiNet-LSTM的High/Low-CR策略
    """

    def __init__(self, cfg):
        super().__init__()

        # 配置参数
        self.T = cfg.model.sequence.T
        D_hi = cfg.model.spatial.latent_dim_high  # 256
        D_lo = cfg.model.spatial.latent_dim_low  # 64

        # CRNet编码器 - 参考原始CRNet设计
        self.enc_hi = CRNetEncoder(
            latent_dim=D_hi,
            res_blocks=cfg.model.spatial.res_blocks + 2,  # 高CR用更深网络
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope
        )

        self.enc_lo = CRNetEncoder(
            latent_dim=D_lo,
            res_blocks=cfg.model.spatial.res_blocks,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope
        )

        # 维度对齐 - 关键改进点
        self.proj_hi2lo = nn.Linear(D_hi, D_lo)

        # LSTM时序建模 - 参考CsiNet-LSTM
        self.temporal = LSTMTemporal(
            dim=D_lo,
            layers=cfg.model.temporal.layers,
            bidirectional=cfg.model.temporal.bidirectional
        )

        # CRNet解码器
        self.decoder = CRNetDecoder(
            latent_dim=D_lo,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope,
            output_activation=cfg.model.spatial.get('output_activation', 'linear')
        )

        # 可学习融合门控（改进版）
        self.fusion_gate = nn.Sequential(
            nn.Linear(D_lo * 2, D_lo // 2),
            nn.ReLU(True),
            nn.Linear(D_lo // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, H, aux=None):
        """
        Args:
            H: [B, T, 2, 32, 32] - 时序CSI矩阵
            aux: [B, M1] - 辅助信息（可选，保持与CsiNet-LSTM兼容）
        """
        B, T = H.size(0), H.size(1)

        # 编码阶段 - 采用CsiNet-LSTM的High/Low-CR策略
        z_raw_list = []
        for t in range(T):
            if t == 0:
                # 第一帧：高维编码
                z = self.enc_hi(H[:, t])
                z = self.proj_hi2lo(z)  # 投影到统一维度
            else:
                # 后续帧：低维编码
                z = self.enc_lo(H[:, t])
            z_raw_list.append(z.unsqueeze(1))

        z_raw_seq = torch.cat(z_raw_list, dim=1)  # [B, T, D_lo]

        # LSTM时序建模
        z_temporal = self.temporal(z_raw_seq)  # [B, T, D_lo]

        # 改进的空间-时序融合
        outputs = []
        for t in range(T):
            z_raw = z_raw_seq[:, t]  # [B, D_lo]
            z_temp = z_temporal[:, t]  # [B, D_lo]

            # 自适应融合权重
            fusion_input = torch.cat([z_raw, z_temp], dim=1)  # [B, D_lo*2]
            alpha = self.fusion_gate(fusion_input)  # [B, 1]

            z_fused = alpha * z_raw + (1 - alpha) * z_temp
            H_hat_t = self.decoder(z_fused)
            outputs.append(H_hat_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, 2, 32, 32]


class CRNetEncoder(nn.Module):
    """
    基于原始CRNet的编码器，增加多分辨率卷积
    """

    def __init__(self, latent_dim, res_blocks=4, channels=32, negative_slope=0.3):
        super().__init__()

        self.negative_slope = negative_slope  # 保存参数
        # 双路径编码器（参考CRNet论文Figure 2）
        self.path1 = nn.Sequential(
            nn.Conv2d(2, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, (1, 9), 1, (0, 4)),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, (9, 1), 1, (4, 0)),
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(2, channels, 3, 1, 1),
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope, True),
        )

        # 残差块序列
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, negative_slope)
            for _ in range(res_blocks)
        ])

        # 下采样：32x32 → 16x16 → 8x8
        self.downsample1 = nn.Conv2d(channels, channels, 3, 2, 1)  # 32→16
        self.downsample2 = nn.Conv2d(channels, channels, 3, 2, 1)  # 16→8
        self.fc = nn.Linear(channels * 8 * 8, latent_dim)

    def forward(self, x):
        # 双路径特征提取
        p1 = self.path1(x)
        p2 = self.path2(x)

        # 特征融合
        x = self.fusion(torch.cat([p1, p2], dim=1))

        # 残差处理
        for block in self.res_blocks:
            x = block(x)

        # 下采样
        x = F.leaky_relu(self.downsample1(x), self.negative_slope, True)  # 32→16
        x = F.leaky_relu(self.downsample2(x), self.negative_slope, True)  # 16→8
        x = self.fc(x.flatten(1))

        return x


class CRNetDecoder(nn.Module):
    """
    基于原始CRNet的解码器，使用CRBlock
    """

    def __init__(self, latent_dim, channels=32, negative_slope=0.3, output_activation='linear'):
        super().__init__()

        self.channels = channels
        self.slope = negative_slope  # 统一管理负斜率
        self.output_activation = output_activation
        self.fc = nn.Linear(latent_dim, channels * 8 * 8)

        # 头部卷积
        self.head_conv = nn.Conv2d(channels, channels, 5, 1, 2)

        # CRBlock x2（参考CRNet论文）
        self.cr_block1 = CRBlock(channels, negative_slope)
        self.cr_block2 = CRBlock(channels, negative_slope)

        # 输出层
        self.output_conv = nn.Conv2d(channels, 2, 3, 1, 1)

    def forward(self, z):
        # 全连接和reshape
        x = self.fc(z).view(-1, self.channels, 8, 8)

        # 上采样到32x32
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

        # 头部处理
        x = F.leaky_relu(self.head_conv(x), self.slope, True)

        # CRBlock处理
        x = self.cr_block1(x)
        x = self.cr_block2(x)

        # 输出激活 - 可配置
        x = self.output_conv(x)

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            return torch.tanh(x)
        else:  # linear
            return x


class CRBlock(nn.Module):
    """
    CRNet的核心CRBlock，实现多分辨率特征提取
    """

    def __init__(self, channels, negative_slope=0.3):
        super().__init__()

        self.slope = negative_slope  # 统一管理

        # 路径1：5x5 + 3x3
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope, True)
        )

        # 路径2：1x9 + 9x1（分解卷积）
        self.path2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 9), 1, (0, 4)),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, (9, 1), 1, (4, 0)),
            nn.LeakyReLU(negative_slope, True)
        )

        # 融合层
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)

        # 拼接和融合
        fused = self.fusion(torch.cat([p1, p2], dim=1))

        # 残差连接
        return F.leaky_relu(x + fused, self.slope, True)


class ResidualBlock(nn.Module):
    """
    基础残差块
    """

    def __init__(self, channels, negative_slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope, True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class LSTMTemporal(nn.Module):
    """
    LSTM时序建模模块 - 修正双向LSTM维度问题
    """

    def __init__(self, dim, layers=3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 如果是双向LSTM，需要降维投影
        if bidirectional:
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = None

    def forward(self, x):
        # x: [B, T, D]
        output, _ = self.lstm(x)  # [B, T, D] or [B, T, 2*D] if bidirectional

        # 如果是双向，投影回原维度
        if self.proj is not None:
            output = self.proj(output)

        return output  # [B, T, D]