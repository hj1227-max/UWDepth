from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

class FeatureModulation(nn.Module):
    """特征调制模块，利用水下参数β和B调整特征"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 调制参数生成网络
        self.modulation_net = nn.Sequential(
            nn.Conv2d(6, channels // 4, 1, 1, 0),  # 输入β和B拼接的6个通道
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, 1, 1, 0),  # 输出scale和bias
        )
        
        # 可选的注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, beta, B):
        batch_size, channels, height, width = x.shape
        
        # 拼接水下参数
        underwater_params = torch.cat([beta, B], dim=1)  # [B, 6, 1, 1]
        
        # 生成调制参数
        modulation_params = self.modulation_net(underwater_params)  # [B, channels*2, 1, 1]
        scale, bias = torch.chunk(modulation_params, 2, dim=1)  # 各[B, channels, 1, 1]
        
        # 应用特征调制
        x_modulated = x * torch.sigmoid(scale) + bias
        
        # 应用通道注意力
        attention_weights = self.channel_attention(x_modulated)
        x_attended = x_modulated * attention_weights
        
        return x_attended
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # 新增：水下参数融合模块
        self.beta_b_fusion = nn.Sequential(
            nn.Conv2d(6, 64, 1, 1, 0),  # 将β和B拼接后融合
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        
        # 特征调制层 - 在每个尺度添加
        self.feature_modulation = nn.ModuleDict()
        for i in range(3):
            self.feature_modulation[str(i)] = FeatureModulation(self.num_ch_dec[i])
        
        # 水下参数增强的跳跃连接调制
        self.skip_modulation = nn.ModuleDict()
        for i in range(3):  # 为可能的跳跃连接层添加调制
            self.skip_modulation[str(i)] = FeatureModulation(self.num_ch_enc[i])

        # 原有decoder结构
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # 改进的深度预测头，结合水下参数
        for s in self.scales:
            # 原有的简单卷积头
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            
            # 新增：增强的深度预测头（可选）
            self.convs[("enhanced_dispconv", s)] = nn.Sequential(
                Conv3x3(self.num_ch_dec[s] + 128, self.num_ch_dec[s] // 2),  # 融合水下特征
                nn.ReLU(inplace=True),
                Conv3x3(self.num_ch_dec[s] // 2, self.num_output_channels)
            )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    def upsample(self, x, scale_factor=2, mode='bilinear'):
        """上采样函数"""
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

    def forward(self, input_features, beta, B):
        """
        Args:
            input_features: 编码器特征列表，不同尺度的特征
            beta: 水下衰减系数 [B, 3, 1, 1]
            B: 水下背景光 [B, 3, 1, 1]
        """
        self.outputs = {}
        
        # 验证输入尺寸
        assert beta.shape[1] == 3 and B.shape[1] == 3, "beta和B应该有3个通道"
        assert beta.shape[2] == 1 and beta.shape[3] == 1, "beta和B应该是空间维度为1x1"
        
        # 融合β和B参数
        beta_b_features = torch.cat([beta, B], dim=1)  # [B, 6, 1, 1]
        beta_b_features = self.beta_b_fusion(beta_b_features)  # [B, 128, 1, 1]
        
        # 初始特征
        x = input_features[-1]
        
        # 将水下参数特征与初始特征融合
        beta_b_expanded = F.interpolate(beta_b_features, size=x.shape[-2:], mode='bilinear')
        x = x + beta_b_expanded  # 初始融合
        
        # 解码过程
        for i in range(2, -1, -1):
            # 第一次上采样卷积
            x = self.convs[("upconv", i, 0)](x)
            x = [self.upsample(x)]
            
            # 跳跃连接
            if self.use_skips and i > 0:
                skip_feature = input_features[i - 1]
                # 对跳跃连接特征进行水下参数调制
                if str(i-1) in self.skip_modulation:
                    skip_feature = self.skip_modulation[str(i-1)](skip_feature, beta, B)
                x += [skip_feature]
                
            # 特征拼接和第二次卷积
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
            # 在关键尺度进行特征调制
            if str(i) in self.feature_modulation:
                x = self.feature_modulation[str(i)](x, beta, B)
            
            # 深度预测
            if i in self.scales:
                # 方法1: 使用原有简单头
                # f = self.upsample(self.convs[("dispconv", i)](x))
                
                # 方法2: 使用增强头（融合水下特征）
                beta_b_current = F.interpolate(beta_b_features, size=x.shape[-2:], mode='bilinear')
                x_augmented = torch.cat([x, beta_b_current], dim=1)
                f = self.upsample(self.convs[("enhanced_dispconv", i)](x_augmented))
                
                self.outputs[("disp", i)] = self.sigmoid(f)

        
        return self.outputs

