import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from DPLTGAN.utils.periodic_activations import SineActivation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma = 5

class Generator(nn.Module):
    def __init__(self, seq_length=150, generated_dim=2, noise_dim=100, embed_dim=10, depth=3,
                 num_heads=3, forward_drop_rate=0.5, attn_drop_rate=0.5,**kwargs):
        super(Generator, self).__init__()
        self.generated_dim = generated_dim
        self.noise_dim = noise_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_timefeature = 6
        # self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        # 第一个是线性层输入大小为latent_dim，输出矩阵大小为self.seq_len *self.embed_dim，比如输入为一个100维的样本，输出为150*10大小的大小量
        # The resulting tensor will have the same batch size as the input and a flattened shape of (self.seq_len, self.embed_dim)
        self.l1 = nn.Linear(self.noise_dim, 2)
        self.LSTM = nn.LSTM(input_size=2, hidden_size=embed_dim, num_layers=1, bias=False, bidirectional=False,
                            batch_first=True)
        # sin cos 位置编码
        # self.pos_embed = nn.Parameter(positional_encoding(self.seq_length, self.embed_dim), requires_grad=False)
        self.SineActivation = SineActivation(6, self.embed_dim)
        self.l2 = nn.Linear(self.embed_dim, self.generated_dim)
        self.LT_TransformerEncoder = LT_TransformerEncoder(depth=depth,
                                                             emb_size=embed_dim,
                                                             drop_p=0.5,
                                                             forward_drop_p=0.5,
                                                             **kwargs)

    # 这句代码定义了一个反卷积层（deconvolution layer），使用 nn.Sequential 将其包装起来。该反卷积层使用 nn.Conv2d 创建，并具有以下参数：

    # self.embed_dim：输入通道数，指定了输入特征的维度。
    # self.channels：输出通道数，指定了反卷积层输出特征的维度，因为有三个特征xyz上的加速度，这里设为3。
    # 1：卷积核的大小，表示卷积核的宽度和高度都为 1。
    # 1：步幅大小，表示卷积操作每次移动的步长为 1。
    # 0：填充大小，表示在输入特征的周围填充 0 的数量为 0。

    def forward(self, z):
        # 首先将采样噪声z=(1000,100)通过线性变换成（1000,1500），然后再转换成（sample_size,seq_len，embed_dim)的大小
        # x=(1000,150,10)
        # x = self.l1(z).view(-1, self.seq_length, self.embed_dim)

        xT_origin = z[:, :, -6:]
        z = z[:, :, :-6]
        x = self.l1(z)
        # 用LSTM代替位置编码
        h0 = torch.randn(1, z.shape[0], self.embed_dim).to(device)
        c0 = torch.randn(1, z.shape[0], self.embed_dim).to(device)
        xh, (h, c) = self.LSTM(x, (h0, c0))

        # 对第0列时间列xT_origin进行time2vec编码
        xT = self.SineActivation(xT_origin)
        feature = self.LT_TransformerEncoder(x, xT, xh)
        # 然后加上位置嵌入pos_embed大小为(1,150,10)
        # x += self.pos_embed
        # 使用Transformer encoder计算attention并得到的编码序列 x=(1000,150,10)
        # feature = self.blocks(x)
        x = self.l2(feature)
        # print('假数据生成完毕，开始转换时间')
        return x

def positional_encoding(seq_length, embed_dim):
    # 创建位置编码矩阵，形状为 (seq_length, embed_dim)
    pos_enc = np.zeros((seq_length, embed_dim))
    # 获取位置索引
    position = np.arange(seq_length)[:, np.newaxis]
    # 计算编码使用的频率
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

    pos_enc[:, 0::2] = np.sin(position * div_term)  # 应用正弦到偶数索引
    pos_enc[:, 1::2] = np.cos(position * div_term[:embed_dim // 2])  # 应用余弦到奇数索引，确保索引不越界

    return torch.tensor(pos_enc, dtype=torch.float32)

class LT_MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 定义输入数据的attention变量
        self.keys_x = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到keys
        self.queries_x = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到queries
        self.values_x = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到values 输入大小为10，
        # 定义隐藏状态的attention变量
        self.keys_h = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到keys
        self.queries_h = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到queries
        # self.values_h = nn.Linear(emb_size, emb_size)# 线性变换层，将输入进行线性变换得到values 输入大小为10，输出大小也为10
        # 定义时间信息的attention变量
        self.keys_t = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到keys
        self.queries_t = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到queries
        # self.values_t = nn.Linear(emb_size, emb_size)# 线性变换层，将输入进行线性变换得到values 输入大小为10，输出大小也为10

        self.att_drop = nn.Dropout(dropout)  # Dropout 层，用于随机置零一部分元素，以防止过拟合，drop概率为0.5
        self.projection = nn.Linear(emb_size, emb_size)  # 线性变换层，将注意力输出进行线性变换

    def forward(self, x: Tensor, xT: Tensor, xh: Tensor, mask: Tensor = None) -> Tensor:
        # 通过线性变换得到 queries、keys 和 values，并进行形状转换
        # "b n (h d)" 表示将原始张量的维度(sample_size,seq_len,num_head*dim)按照（sample_size,head_num,seq_len,dim）进行重新排列。
        # x大小为(1000,150,10),queries的大小为（1000,5,150,2) 但是前面说embed_size=10想不通为啥要拆成小的维度？？？
        seq_len = x.shape[1]
        queries_x = rearrange(self.queries_x(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys_x = rearrange(self.keys_x(x), "b n (h d) -> b h n d", h=self.num_heads)
        values_x = rearrange(self.values_x(x), "b n (h d) -> b h n d", h=self.num_heads)

        queries_h = rearrange(self.queries_h(xh), "b n (h d) -> b h n d", h=self.num_heads)
        keys_h = rearrange(self.keys_h(xh), "b n (h d) -> b h n d", h=self.num_heads)
        # values_h = rearrange(self.values_h(xh), "b n (h d) -> b h n d", h=self.num_heads)

        queries_t = rearrange(self.queries_t(xT), "b n (h d) -> b h n d", h=self.num_heads)
        keys_t = rearrange(self.keys_t(xT), "b n (h d) -> b h n d", h=self.num_heads)
        # values_t = rearrange(self.values_t(xT), "b n (h d) -> b h n d", h=self.num_heads)
        # 计算注意力分数 energy，使用 einsum 进行张量运算 torch.einsum(equation, *operands): 这个函数实现了爱因斯坦求和约定。它接受一个方程字符串 equation 和多个操作数 operands，并根据方程字符串进行张量乘法和求和运算。
        # 方程字符串 'bhqd, bhkd -> bhqk' 表示对两个张量进行乘法运算，其中 b 是批量维度，h 是注意力头维度，q 和 k 是查询和键的维度。
        # energy大小为（1000,5,150,150)代表每个位置和其它位置的注意力权重大小矩阵
        energy_x = torch.einsum('bhqd, bhkd -> bhqk', queries_x, keys_x)  # batch, num_heads, query_len,
        energy_h = torch.einsum('bhqd, bhkd -> bhqk', queries_h, keys_h)
        energy_t = torch.einsum('bhqd, bhkd -> bhqk', queries_t, keys_t)
        #  将根据x与时间编码和隐藏状态计算的权重大小矩阵相加并放缩
        energy = torch.add(energy_x, energy_h)
        energy_new = torch.add(energy, energy_t)
        energy_new = torch.div(energy_new, 3)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy_new.mask_fill(~mask, fill_value)
        # 对注意力进行缩放，避免注意力值太大
        scaling = self.emb_size ** (1 / 2)
        # 对注意力求softmax结果将其限制在0-1 att大小为（1000,5,150,150)
        # dim=-1对最后一个维度进行softmax运算,att每一列注意力权重之和为1
        att = energy_new / scaling
        # 加上顺序约束系数
        # 制造顺序差矩阵
        seqrow = np.arange(seq_len)
        seqrow = np.repeat(seqrow, seq_len)
        arrrow = seqrow.reshape(seq_len, seq_len)
        seqcol = np.arange(seq_len)
        arrcol = np.tile(seqcol, (seq_len, 1))
        subarr = arrrow - arrcol
        gammaarr = np.where(subarr > 0, 0, subarr)
        gammaarr = np.exp(-np.square(gammaarr) / 5)
        # 只取其对角线以上部分的系数，因为对角线以上部分表示之前时间对该节点的影响
        gammaarr = np.triu(gammaarr, k=1)
        gammaarr = gammaarr + 1
        gamma = torch.from_numpy(gammaarr).to(device)
        # 对列元素进行softmax操作，使其列元素之和为1
        # gamma的第i列代表了各时间点对第i点的顺序约束系数
        # gamma = F.softmax(gamma, dim=-1)
        gamma = gamma.unsqueeze(0).unsqueeze(0)
        # 让每个attention权重乘以一个系数，使得过去时间段的轨迹点对当前时间点数据的注意力值增大
        newatt = att * gamma
        # newatt的第i行代表了各时间点对第i点的注意力权重
        newatt = F.softmax(newatt, dim=-1)
        att = self.att_drop(newatt)
        att = att.float()
        # 'a' 表示查询长度维度，表示查询张量的长度（可以理解为输入序列的长度）。
        # 'l' 表示键长度维度，表示键张量的长度（同样可以理解为输入序列的长度）。
        # 'v' 表示值维度，表示值张量的长度。
        # 将注意力和values值相乘，计算最终结果
        # out的大小为（1000,5,150,2）att=(1000,5,150,150),values=(1000,5,150,2)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values_x).to(device)
        # out=(1000,150,10)
        out = rearrange(out, "b h n d -> b n (h d)")
        # 单纯进行线性变换，输入多少维度输出还是多少维度
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # res大小为（sample_size,seq_len,embed_dim)
        res = x
        x = self.fn(x, **kwargs)
        x_new = x + res
        return x_new


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class LSTMModel(nn.Module):
    """
    循环神经网络模型，作为ATPS实现中，generator和discriminator的父类，
    generator和discriminator相同的层已经在这个类里面写好了
    generator和discriminator只需要继承这个类，然后加上自己的输出层就好
    """

    def __init__(self, feature_size, noise_size, batch_size, hidden_size, **kwargs):
        """
        :param feature_size: 每个数据的维度，比如输入如果是(longitude, latitude)，本参数即为2
        :param noise_size: 噪声的维度，
        :param batch_size: 一次训练几条数据
        :param hidden_size: GRU隐藏层的维度
        :param kwargs:
        """
        super(LSTMModel, self).__init__(**kwargs)

        self.feature_size = feature_size
        self.num_hiddens = self.rnn.hidden_size
        self.batch_size = batch_size
        print("self.feature_size:", self.feature_size)
        print("self.num_hiddens:", self.num_hiddens)

        # 输入维度是时间序列中每个点的维度(longitude, latitude, day of week, hour, minute)
        # 输出维度是时间序列中每个点的维度
        self.embedding = nn.Linear(feature_size, feature_size)

        self.rnn = nn.LSTM(feature_size, hidden_size)

    def begin_state(self, batch_size, device):
        return torch.zeros(
            (self.rnn.num_layers, batch_size, self.num_hiddens),
            device=device
        )


class LT_TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels=3,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__()
        # self.ResidualAdd = ResidualAdd()
        self.Linear = nn.Linear(2, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)

        self.LT_MultiHeadAttention = LT_MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.FeedForwardBlock = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)

    def forward(self, x: Tensor, xh: Tensor, xT: Tensor):
        # 先对输入数据进行维度变换
        x = self.Linear(x)
        # 编码器分为两部分，第一部分是attention，第二部分是feedforward变换
        xL = self.LayerNorm(x)
        xH = self.LayerNorm(xh)
        xT = self.LayerNorm(xT)
        output = self.LT_MultiHeadAttention(xL, xT, xH)
        output = self.dropout(output)
        # 残差连接，x和第一部分的输出结果相加
        x_new = x + output

        # 第二部分
        x1 = self.LayerNorm(x_new)
        x1 = self.FeedForwardBlock(x1)
        x1 = self.dropout(x1)
        # 残差连接，第二部分输入和第二部分输出结果相加
        x_new = x_new + x1
        return x_new


class LT_TransformerEncoder(nn.Module):
    def __init__(self, depth=1, **kwargs):
        super().__init__()
        self.depth = depth
        self.block = LT_TransformerEncoderBlock(**kwargs)

    def forward(self, x: Tensor, xh: Tensor, xT: Tensor):
        # *[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)
        # for i in range(self.depth):
        return self.block(x, xh, xT)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
            nn.Sigmoid()
        )


class PatchEmbedding_Linear(nn.Module):
    # what are the proper parameters set here?
    # 这是类的构造函数，用于初始化类的参数。其中，in_channels表示输入图像的通道数，patch_size表示每个分块的大小，emb_size表示嵌入的维度，seq_length表示序列的长度
    def __init__(self, in_channels=21, patch_size=16, emb_size=100, seq_length=1024):
        # self.patch_size = patch_size
        super().__init__()
        # change the conv2d parameters here
        # self.projection：这是一个包含两个模块的序列。第一个模块是Rearrange，用于将输入图像进行分块并重排维度。
        # 第二个模块是一个线性层(nn.Linear)，用于将分块后的图像进行嵌入。patch_size*in_channels表示将每个分块的像素值展平为一个向量
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            nn.Linear(patch_size * in_channels, emb_size)
        )
        # self.cls_token：这是一个可学习的参数，表示类别标记的嵌入向量。在处理图像序列时，通常在输入序列前加入一个类别标记。
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions：这是一个可学习的参数，表示位置嵌入向量。通过将位置嵌入向量与输入的每个位置相加，可以引入位置信息
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # x = self.projection(x)：将输入图像进行分块嵌入，得到嵌入后的张量。
        x = self.projection(x)
        # 将类别标记的嵌入向量复制多次，并与输入张量的第一维度进行拼接，以添加类别标记到输入序列的开头。
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        # 通过将位置嵌入向量与输入的每个位置相加，可以引入位置信息
        x_encode = torch.add(x, self.positions)
        return x



class OriDis_MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 定义输入数据的attention变量
        self.keys = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到keys
        self.queries = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到queries
        self.values = nn.Linear(emb_size, emb_size)  # 线性变换层，将输入进行线性变换得到values 输入大小为10，

        self.att_drop = nn.Dropout(dropout)  # Dropout 层，用于随机置零一部分元素，以防止过拟合，drop概率为0.5
        self.projection = nn.Linear(emb_size, emb_size)  # 线性变换层，将注意力输出进行线性变换

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 通过线性变换得到 queries、keys 和 values，并进行形状转换
        # "b n (h d)" 表示将原始张量的维度(sample_size,seq_len,num_head*dim)按照（sample_size,head_num,seq_len,dim）进行重新排列。
        # x大小为(1000,150,10),queries的大小为（1000,5,150,2) 但是前面说embed_size=10想不通为啥要拆成小的维度？？？
        seq_len = x.shape[1]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # 计算注意力分数 energy，使用 einsum 进行张量运算 torch.einsum(equation, *operands): 这个函数实现了爱因斯坦求和约定。它接受一个方程字符串 equation 和多个操作数 operands，并根据方程字符串进行张量乘法和求和运算。
        # 方程字符串 'bhqd, bhkd -> bhqk' 表示对两个张量进行乘法运算，其中 b 是批量维度，h 是注意力头维度，q 和 k 是查询和键的维度。
        # energy大小为（1000,5,150,150)代表每个位置和其它位置的注意力权重大小矩阵
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        # 对注意力进行缩放，避免注意力值太大
        scaling = self.emb_size ** (1 / 2)
        # 对注意力求softmax结果将其限制在0-1 att大小为（1000,5,150,150)
        # dim=-1对最后一个维度进行softmax运算,att每一列注意力权重之和为1
        att = energy / scaling
        # 让每个attention权重乘以一个系数，使得过去时间段的轨迹点对当前时间点数据的注意力值增大
        newatt = att
        # newatt的第i行代表了各时间点对第i点的注意力权重
        newatt = F.softmax(newatt, dim=-1)
        att = self.att_drop(newatt)
        att = att.float()
        # 'a' 表示查询长度维度，表示查询张量的长度（可以理解为输入序列的长度）。
        # 'l' 表示键长度维度，表示键张量的长度（同样可以理解为输入序列的长度）。
        # 'v' 表示值维度，表示值张量的长度。
        # 将注意力和values值相乘，计算最终结果
        # out的大小为（1000,5,150,2）att=(1000,5,150,150),values=(1000,5,150,2)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out=(1000,150,10)
        out = rearrange(out, "b h n d -> b n (h d)")
        # 单纯进行线性变换，输入多少维度输出还是多少维度
        out = self.projection(out)
        return out

class OriDis_TransformerEncoder(nn.Module):
    def __init__(self, depth=1, **kwargs):
        super().__init__()
        self.depth = depth
        self.block = OriDis_TransformerEncoderBlock(**kwargs)

    def forward(self, x: Tensor):
        # *[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)
        # for i in range(self.depth):
        return self.block(x)
class OriDis_TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels=3,
                 emb_size=100,
                 num_heads=2,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__()
        # self.ResidualAdd = ResidualAdd()
        self.Linear = nn.Linear(2, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)

        self.Dis_MultiHeadAttention = OriDis_MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.FeedForwardBlock = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)

    def forward(self, x: Tensor):
        # 先对输入数据进行维度变换
        x = self.Linear(x)
        # 编码器分为两部分，第一部分是attention，第二部分是feedforward变换
        xL = self.LayerNorm(x)
        # xH = self.LayerNorm(xh)
        # xT = self.LayerNorm(xT)
        output = self.Dis_MultiHeadAttention(xL)
        output = self.dropout(output)
        # 残差连接，x和第一部分的输出结果相加
        x_new = x + output

        # 第二部分
        x1 = self.LayerNorm(x_new)
        x1 = self.FeedForwardBlock(x1)
        x1 = self.dropout(x1)
        # 残差连接，第二部分输入和第二部分输出结果相加
        x_new = x_new + x1
        return x_new
class OriTrans_Discriminator(nn.Module):
    def __init__(self,
                 in_channels=8,
                 emb_size=50,
                 seq_length=20,
                 depth=1,
                 n_classes=1,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.emb_size = emb_size
        self.seq_length = seq_length
        self.depth = depth
        self.n_classes = n_classes
        # self.SineActivation = SineActivation(6, self.emb_size)
        self.LSTM = nn.LSTM(input_size=2, hidden_size=emb_size, num_layers=1, bias=False, bidirectional=False,
                            batch_first=True)
        self.pos_embed = nn.Parameter(positional_encoding(self.seq_length, 2), requires_grad=False)
        self.Dis_TransformerEncoder = OriDis_TransformerEncoder(depth=depth,
                                                             emb_size=emb_size,
                                                             drop_p=0.5,
                                                             forward_drop_p=0.5,
                                                             **kwargs)
        self.ClassificationHead = ClassificationHead(emb_size, n_classes)

        # 这句代码定义了一个反卷积层（deconvolution layer），使用 nn.Sequential 将其包装起来。该反卷积层使用 nn.Conv2d 创建，并具有以下参数：

        # self.embed_dim：输入通道数，指定了输入特征的维度。
        # self.channels：输出通道数，指定了反卷积层输出特征的维度，因为有三个特征xyz上的加速度，这里设为3。

    def forward(self, x):
        # 首先通过LSTM获取输入序列的隐藏状态
        # x=(1000,150,10)    LSTM要输入的为形状为(seq_length, batch_size, input_size)或者(batch_size, seq_length, input_size)
        # * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
        #           :math:`(D * \text{num\_layers}, N, H_{out})` containing the
        #           initial hidden state for each element in the input sequence.
        #           Defaults to zeros if (h_0, c_0) is not provided.
        # * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
        #           :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
        #           initial cell state for each element in the input sequence.
        #           Defaults to zeros if (h_0, c_0) is not provided.
        # 传入的x数据为一个大小为(batch_size,seq_len,num_features)的数据，其中第一维参数为转换成timestamp类型的时间戳数据
        #  第二维和第三维数据分别为坐标的经纬度
        x1 = x[:, :, :2]
        xh_origin = x1+ self.pos_embed
        # 使用Enhanced Transformer提取真实轨迹与生成轨迹特征
        feature = self.Dis_TransformerEncoder(xh_origin)
        y = self.ClassificationHead(feature)
        return y

if __name__ == '__main__':
    x = torch.randn(1000, 150, 3)
    d = Discriminator()
    y = d(x)
    print("y:", y)
    print(y)



