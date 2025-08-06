# models/crlstmnet.py - Core model implementation (Corrected version)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRLSTMNet(nn.Module):
    """
    Lightweight version that fuses CRNet spatial feature extraction and LSTM temporal modeling
    References the High/Low-CR strategy from CsiNet-LSTM
    """

    def __init__(self, cfg):
        super().__init__()

        # Configuration parameters
        self.T = cfg.model.sequence.T
        D_hi = cfg.model.spatial.latent_dim_high  # 256
        D_lo = cfg.model.spatial.latent_dim_low  # 64

        # CRNet encoders - based on original CRNet design
        self.enc_hi = CRNetEncoder(
            latent_dim=D_hi,
            res_blocks=cfg.model.spatial.res_blocks + 2,  # Deeper network for high CR
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope
        )

        self.enc_lo = CRNetEncoder(
            latent_dim=D_lo,
            res_blocks=cfg.model.spatial.res_blocks,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope
        )

        # Dimension alignment - key improvement point
        self.proj_hi2lo = nn.Linear(D_hi, D_lo)

        # LSTM temporal modeling - based on CsiNet-LSTM
        self.temporal = LSTMTemporal(
            dim=D_lo,
            layers=cfg.model.temporal.layers,
            bidirectional=cfg.model.temporal.bidirectional
        )

        # CRNet decoder
        self.decoder = CRNetDecoder(
            latent_dim=D_lo,
            channels=cfg.model.spatial.channels,
            negative_slope=cfg.model.spatial.negative_slope,
            output_activation=cfg.model.spatial.get('output_activation', 'linear')
        )

        # Learnable fusion gate (improved version)
        self.fusion_gate = nn.Sequential(
            nn.Linear(D_lo * 2, D_lo // 2),
            nn.ReLU(True),
            nn.Linear(D_lo // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, H, aux=None):
        """
        Args:
            H: [B, T, 2, 32, 32] - Temporal CSI matrices
            aux: [B, M1] - Auxiliary information (optional, for CsiNet-LSTM compatibility)
        """
        B, T = H.size(0), H.size(1)

        # Encoding stage - using CsiNet-LSTM's High/Low-CR strategy
        z_raw_list = []
        for t in range(T):
            if t == 0:
                # First frame: high-dimensional encoding
                z = self.enc_hi(H[:, t])
                z = self.proj_hi2lo(z)  # Project to unified dimension
            else:
                # Subsequent frames: low-dimensional encoding
                z = self.enc_lo(H[:, t])
            z_raw_list.append(z.unsqueeze(1))

        z_raw_seq = torch.cat(z_raw_list, dim=1)  # [B, T, D_lo]

        # LSTM temporal modeling
        z_temporal = self.temporal(z_raw_seq)  # [B, T, D_lo]

        # Improved spatial-temporal fusion
        outputs = []
        for t in range(T):
            z_raw = z_raw_seq[:, t]  # [B, D_lo]
            z_temp = z_temporal[:, t]  # [B, D_lo]

            # Adaptive fusion weights
            fusion_input = torch.cat([z_raw, z_temp], dim=1)  # [B, D_lo*2]
            alpha = self.fusion_gate(fusion_input)  # [B, 1]

            z_fused = alpha * z_raw + (1 - alpha) * z_temp
            H_hat_t = self.decoder(z_fused)
            outputs.append(H_hat_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, 2, 32, 32]


class CRNetEncoder(nn.Module):
    """
    Encoder based on original CRNet, with added multi-resolution convolution
    """

    def __init__(self, latent_dim, res_blocks=4, channels=32, negative_slope=0.3):
        super().__init__()

        self.negative_slope = negative_slope  # Save parameter
        # Dual-path encoder (reference CRNet paper Figure 2)
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

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope, True),
        )

        # Residual block sequence
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, negative_slope)
            for _ in range(res_blocks)
        ])

        # Downsampling: 32x32 → 16x16 → 8x8
        self.downsample1 = nn.Conv2d(channels, channels, 3, 2, 1)  # 32→16
        self.downsample2 = nn.Conv2d(channels, channels, 3, 2, 1)  # 16→8
        self.fc = nn.Linear(channels * 8 * 8, latent_dim)

    def forward(self, x):
        # Dual-path feature extraction
        p1 = self.path1(x)
        p2 = self.path2(x)

        # Feature fusion
        x = self.fusion(torch.cat([p1, p2], dim=1))

        # Residual processing
        for block in self.res_blocks:
            x = block(x)

        # Downsampling
        x = F.leaky_relu(self.downsample1(x), self.negative_slope, True)  # 32→16
        x = F.leaky_relu(self.downsample2(x), self.negative_slope, True)  # 16→8
        x = self.fc(x.flatten(1))

        return x


class CRNetDecoder(nn.Module):
    """
    Decoder based on original CRNet, using CRBlocks
    """

    def __init__(self, latent_dim, channels=32, negative_slope=0.3, output_activation='linear'):
        super().__init__()

        self.channels = channels
        self.slope = negative_slope  # Unified management of negative slope
        self.output_activation = output_activation
        self.fc = nn.Linear(latent_dim, channels * 8 * 8)

        # Head convolution
        self.head_conv = nn.Conv2d(channels, channels, 5, 1, 2)

        # CRBlock x2 (reference CRNet paper)
        self.cr_block1 = CRBlock(channels, negative_slope)
        self.cr_block2 = CRBlock(channels, negative_slope)

        # Output layer
        self.output_conv = nn.Conv2d(channels, 2, 3, 1, 1)

    def forward(self, z):
        # Fully connected and reshape
        x = self.fc(z).view(-1, self.channels, 8, 8)

        # Upsample to 32x32
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

        # Head processing
        x = F.leaky_relu(self.head_conv(x), self.slope, True)

        # CRBlock processing
        x = self.cr_block1(x)
        x = self.cr_block2(x)

        # Output activation - configurable
        x = self.output_conv(x)

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            return torch.tanh(x)
        else:  # linear
            return x


class CRBlock(nn.Module):
    """
    Core CRBlock of CRNet, implements multi-resolution feature extraction
    """

    def __init__(self, channels, negative_slope=0.3):
        super().__init__()

        self.slope = negative_slope  # Unified management

        # Path 1: 5x5 + 3x3
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope, True)
        )

        # Path 2: 1x9 + 9x1 (factorized convolution)
        self.path2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 9), 1, (0, 4)),
            nn.LeakyReLU(negative_slope, True),
            nn.Conv2d(channels, channels, (9, 1), 1, (4, 0)),
            nn.LeakyReLU(negative_slope, True)
        )

        # Fusion layer
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)

        # Concatenation and fusion
        fused = self.fusion(torch.cat([p1, p2], dim=1))

        # Residual connection
        return F.leaky_relu(x + fused, self.slope, True)


class ResidualBlock(nn.Module):
    """
    Basic residual block
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
    LSTM temporal modeling module - Fixed bidirectional LSTM dimension issue
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

        # If bidirectional LSTM, need dimensionality reduction projection
        if bidirectional:
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = None

    def forward(self, x):
        # x: [B, T, D]
        output, _ = self.lstm(x)  # [B, T, D] or [B, T, 2*D] if bidirectional

        # If bidirectional, project back to original dimension
        if self.proj is not None:
            output = self.proj(output)

        return output  # [B, T, D]