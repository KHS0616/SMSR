"""
SMSR Model Code

Writer : KHS0616
Last Update : 2021-11-03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    # 0이 하나라도 있으면 다시 생성
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    # torch.log -> log연산 수행 후 값 반환
    gumbels = -(-gumbels.log()).log()

    # 생성한 gumbels noise tensor, 입력 x를 합쳐서 sofrmax 연산 수행
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class SMSR(nn.Module):
    """ SMSR 네트워크 클래스 """
    def __init__(self):
        super(SMSR, self).__init__()
        # 옵션 설정
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        rgb_range = 255

        self.scale = 2

        # DIV2K 데이터 셋을 위한 RGB 정규화 수치 저장
        # 해당 수치를 이용하여 MeanShift 설정
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # Head Layer 설정
        # 구성은 3x3 conv + ReLU + 3x3 conv
        modules_head = [nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size//2), bias=True),
                        nn.ReLU(True),
                        nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=True)]

        # Body Layer 설정
        # 구성은 SMM 블록 5번 반복
        modules_body = [SMM(n_feats, n_feats, kernel_size) \
                        for _ in range(5)]

        # Collect Layer 설정
        # 구성은 1x1 conv + ReLU + 3x3 conv
        self.collect = nn.Sequential(
            nn.Conv2d(64*5, 64, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        # Tail Layer 설정
        # PixelShuffle 연산을 통해 Upsample
        modules_tail = [
            nn.Conv2d(n_feats, n_colors*self.scale*self.scale, 3, 1, 1),
            nn.PixelShuffle(self.scale),
        ]

        # 변수로 각 Layer 등록
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # 이미지 정규화 및 Head 통과
        x0 = self.sub_mean(x)
        x = self.head(x0)

        if self.training:
            sparsity = []
            out_fea = []
            fea = x
            
            # 5번의 SMM 연산 수행
            for i in range(5):                
                fea, _spa_mask, _ch_mask = self.body[i](fea)
                out_fea.append(fea)
                sparsity.append(_spa_mask * _ch_mask[..., 1].view(1, -1, 1, 1) + torch.ones_like(_spa_mask) * _ch_mask[..., 0].view(1, -1, 1, 1))

            # SMM 통과 이전 결과와 통과 이후 결과 합치기
            out_fea = self.collect(torch.cat(out_fea, 1)) + x

            # sparsity concat 연산으로 합치기
            sparsity = torch.cat(sparsity, 0)

            # Upsamole 연산 수행
            x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)

            # 이미지 정규화 해제
            x = self.add_mean(x)

            return [x, sparsity]

        if not self.training:
            out_fea = []
            fea = x
            for i in range(5):
                fea = self.body[i](fea)
                out_fea.append(fea)
            out_fea = self.collect(torch.cat(out_fea, 1)) + x

            x = self.tail(out_fea) + F.interpolate(x0, scale_factor=self.scale, mode='bicubic', align_corners=False)
            x = self.add_mean(x)

            return x

class MeanShift(nn.Conv2d):
    """ 이미지 정규화 네트워크 """
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y

class SMM(nn.Module):
    """ SMM 블록 네트워크 클래스 """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SMM, self).__init__()

        # sparse mask generator
        # 모래시계 모양 블록
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels // 4, 2, 3, 2, 1, output_padding=1),
        )

        # body
        self.body = SMB(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=4)

        # CA layer
        # Channel Attention
        self.ca = CALayer(out_channels)

        self.tau = 1

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        if self.training:
            # spa 마스크 생성하고 gumbel_softmax 진행
            spa_mask = self.spa_mask(x)
            spa_mask = gumbel_softmax(spa_mask, 1, self.tau)

            # Body 부분을 통해 channel mask 생성 및 이미지 연산
            out, ch_mask = self.body([x, spa_mask[:, 1:, ...]])
            out = self.ca(out) + x

            return out, spa_mask[:, 1:, ...], ch_mask

        if not self.training:
            spa_mask = self.spa_mask(x)
            spa_mask = (spa_mask[:, 1:, ...] > spa_mask[:, :1, ...]).float()

            out = self.body([x, spa_mask])
            out = self.ca(out) + x

            return out

class SMB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=4):
        super(SMB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.tau = 1
        self.relu = nn.ReLU(True)

        # channels mask 초기화
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))

        # body - sparse Conv 수행과정에서 Feature Map 연산 부분
        body = []

        # 입력받은 Feature Map을 합성곱 연산 수행
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers-1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)

        # collect
        self.collect = nn.Conv2d(out_channels*self.n_layers, out_channels, 1, 1, 0)

    def _update_tau(self, tau):
        self.tau = tau

    def _prepare(self):
        # channel mask
        ch_mask = self.ch_mask.softmax(3).round()
        self.ch_mask_round = ch_mask

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers):
            if i == 0:
                self.d_in_num.append(self.in_channels)
                self.s_in_num.append(0)
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))
            else:
                self.d_in_num.append(int(ch_mask[0, :, i-1, 0].sum(0)))
                self.s_in_num.append(int(ch_mask[0, :, i-1, 1].sum(0)))
                self.d_out_num.append(int(ch_mask[0, :, i, 0].sum(0)))
                self.s_out_num.append(int(ch_mask[0, :, i, 1].sum(0)))

        # kernel split
        kernel_d2d = []
        kernel_d2s = []
        kernel_s = []

        # 커널 스플릿
        for i in range(self.n_layers):
            if i == 0:
                kernel_s.append([])
                if self.d_out_num[i] > 0:
                    kernel_d2d.append(self.body[i].weight[ch_mask[0, :, i, 0]==1, ...].view(self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.s_out_num[i] > 0:
                    kernel_d2s.append(self.body[i].weight[ch_mask[0, :, i, 1]==1, ...].view(self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
            else:
                if self.d_in_num[i] > 0 and self.d_out_num[i] > 0:
                    kernel_d2d.append(
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i-1, 0] == 1, ...].view(self.d_out_num[i], -1))
                else:
                    kernel_d2d.append([])
                if self.d_in_num[i] > 0 and self.s_out_num[i] > 0:
                    kernel_d2s.append(
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i-1, 0] == 1, ...].view(self.s_out_num[i], -1))
                else:
                    kernel_d2s.append([])
                if self.s_in_num[i] > 0:
                    kernel_s.append(torch.cat((
                        self.body[i].weight[ch_mask[0, :, i, 0] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...],
                        self.body[i].weight[ch_mask[0, :, i, 1] == 1, ...][:, ch_mask[0, :, i - 1, 1] == 1, ...]),
                        0).view(self.d_out_num[i]+self.s_out_num[i], -1))
                else:
                    kernel_s.append([])

        # the last 1x1 conv
        ch_mask = ch_mask[0, ...].transpose(1, 0).contiguous().view(-1, 2)
        self.d_in_num.append(int(ch_mask[:, 0].sum(0)))
        self.s_in_num.append(int(ch_mask[:, 1].sum(0)))
        self.d_out_num.append(self.out_channels)
        self.s_out_num.append(0)

        kernel_d2d.append(self.collect.weight[:, ch_mask[..., 0] == 1, ...].squeeze())
        kernel_d2s.append([])
        kernel_s.append(self.collect.weight[:, ch_mask[..., 1] == 1, ...].squeeze())

        self.kernel_d2d = kernel_d2d
        self.kernel_d2s = kernel_d2s
        self.kernel_s = kernel_s
        self.bias = self.collect.bias

    def _generate_indices(self):
        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        # torch.nonzero - 0이 아닌 요소의 인덱스를 반환
        mask_indices = torch.nonzero(self.spa_mask.squeeze())

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        # zero padding 실시
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)

    def _sparse_conv(self, fea_dense, fea_sparse, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.d_out_num[index] > 0:
                # dense to dense
                if k > 1:
                    fea_col = F.unfold(fea_dense, k, stride=1, padding=(k-1) // 2).squeeze(0)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))
                else:
                    fea_col = fea_dense.view(self.d_in_num[index], -1)
                    fea_d2d = torch.mm(self.kernel_d2d[index].view(self.d_out_num[index], -1), fea_col)
                    fea_d2d = fea_d2d.view(1, self.d_out_num[index], fea_dense.size(2), fea_dense.size(3))

            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel_d2s[index], self._mask_select(fea_dense, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel_s[index], fea_sparse)
            else:
                fea_s2ds = torch.mm(self.kernel_s[index], F.pad(fea_sparse, [1,0,0,0])[:, self.idx_s2s].view(self.s_in_num[index] * k * k, -1))

        # fusion
        if self.d_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_d2d[0, :, self.h_idx_1x1, self.w_idx_1x1] += fea_s2ds[:self.d_out_num[index], :]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
                fea_d[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds[:self.d_out_num[index], :]
        else:
            fea_d = None

        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                if self.s_in_num[index] > 0:
                    fea_s = fea_d2s + fea_s2ds[ -self.s_out_num[index]:, :]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[-self.s_out_num[index]:, :]
        else:
            fea_s = None

        # add bias (bias is only used in the last 1x1 conv in our SMB for simplicity)
        if index == 4:
            fea_d += self.bias.view(1, -1, 1, 1)

        return fea_d, fea_s

    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature (B, C ,H, W) ;
        x[1]: sparse mask (B, 1, H, W)
        '''
        if self.training:
            # sparse, channel 마스크 저장
            spa_mask = x[1]
            ch_mask = gumbel_softmax(self.ch_mask, 3, self.tau)

            # Train sparse Conv 연산 수행
            out = []
            fea = x[0]
            for i in range(self.n_layers):
                # 첫 layer 에서는 Conv 연산 수행 후 channel mask를 2등분하여 곱셈연산만 수행
                if i == 0:
                    fea = self.body[i](fea)
                    fea = fea * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea * ch_mask[:, :, i:i + 1, :1]

                # 두 번째 layer부터 channel mask를 2등분하여 각각 곱셈연산을 수행하고 다시 한번 4등분하여 곱셈 후 덧셈연산수행
                else:
                    # Fd, Fs 각각하고 channel mask 곱하고 conv 연산
                    fea_d = self.body[i](fea * ch_mask[:, :, i - 1:i, :1])
                    fea_s = self.body[i](fea * ch_mask[:, :, i - 1:i, 1:])

                    # 각각의 결과를 곱하고 최종적으로 합치기
                    fea = fea_d * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_d * ch_mask[:, :, i:i + 1, :1] + \
                          fea_s * ch_mask[:, :, i:i + 1, 1:] * spa_mask + fea_s * ch_mask[:, :, i:i + 1, :1] * spa_mask

                # 최종 결과물 활성화함수 통과
                fea = self.relu(fea)
                out.append(fea)

            # conv연산으로 결과 출력
            out = self.collect(torch.cat(out, 1))

            return out, ch_mask

        if not self.training:
            self.spa_mask = x[1]

            # generate indices
            self._generate_indices()

            # sparse conv
            fea_d = x[0]
            fea_s = None
            fea_dense = []
            fea_sparse = []
            for i in range(self.n_layers):
                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, k=3, index=i)
                if fea_d is not None:
                    fea_dense.append(self.relu(fea_d))
                if fea_s is not None:
                    fea_sparse.append(self.relu(fea_s))

            # 1x1 conv
            fea_dense = torch.cat(fea_dense, 1)
            fea_sparse = torch.cat(fea_sparse, 0)
            out, _ = self._sparse_conv(fea_dense, fea_sparse, k=1, index=self.n_layers)

            return out