import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from utils.periodic_activations import SineActivation

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

        # The resulting tensor will have the same batch size as the input and a flattened shape of (self.seq_len, self.embed_dim)
        self.l1 = nn.Linear(self.noise_dim, 2)
        self.LSTM = nn.LSTM(input_size=2, hidden_size=embed_dim, num_layers=1, bias=False, bidirectional=False,
                            batch_first=True)

        self.SineActivation = SineActivation(6, self.embed_dim)
        self.l2 = nn.Linear(self.embed_dim, self.generated_dim)
        self.LT_TransformerEncoder = LT_TransformerEncoder(depth=depth,
                                                             emb_size=embed_dim,
                                                             drop_p=0.5,
                                                             forward_drop_p=0.5,
                                                             **kwargs)



    def forward(self, z):

        xT_origin = z[:, :, -6:]
        z = z[:, :, :-6]
        x = self.l1(z)

        h0 = torch.randn(1, z.shape[0], self.embed_dim).to(device)
        c0 = torch.randn(1, z.shape[0], self.embed_dim).to(device)
        xh, (h, c) = self.LSTM(x, (h0, c0))
        xT = self.SineActivation(xT_origin)
        feature = self.LT_TransformerEncoder(x, xT, xh)
        x = self.l2(feature)
        return x

def positional_encoding(seq_length, embed_dim):
    pos_enc = np.zeros((seq_length, embed_dim))
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term[:embed_dim // 2])

    return torch.tensor(pos_enc, dtype=torch.float32)

class LT_MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys_x = nn.Linear(emb_size, emb_size)
        self.queries_x = nn.Linear(emb_size, emb_size)
        self.values_x = nn.Linear(emb_size, emb_size)
        self.keys_h = nn.Linear(emb_size, emb_size)
        self.queries_h = nn.Linear(emb_size, emb_size)
        self.keys_t = nn.Linear(emb_size, emb_size)
        self.queries_t = nn.Linear(emb_size, emb_size)
        # self.values_t = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, xT: Tensor, xh: Tensor, mask: Tensor = None) -> Tensor:
        seq_len = x.shape[1]
        queries_x = rearrange(self.queries_x(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys_x = rearrange(self.keys_x(x), "b n (h d) -> b h n d", h=self.num_heads)
        values_x = rearrange(self.values_x(x), "b n (h d) -> b h n d", h=self.num_heads)

        queries_h = rearrange(self.queries_h(xh), "b n (h d) -> b h n d", h=self.num_heads)
        keys_h = rearrange(self.keys_h(xh), "b n (h d) -> b h n d", h=self.num_heads)
        # values_h = rearrange(self.values_h(xh), "b n (h d) -> b h n d", h=self.num_heads)

        queries_t = rearrange(self.queries_t(xT), "b n (h d) -> b h n d", h=self.num_heads)
        keys_t = rearrange(self.keys_t(xT), "b n (h d) -> b h n d", h=self.num_heads)
        energy_x = torch.einsum('bhqd, bhkd -> bhqk', queries_x, keys_x)  # batch, num_heads, query_len,
        energy_h = torch.einsum('bhqd, bhkd -> bhqk', queries_h, keys_h)
        energy_t = torch.einsum('bhqd, bhkd -> bhqk', queries_t, keys_t)
        energy = torch.add(energy_x, energy_h)
        energy_new = torch.add(energy, energy_t)
        energy_new = torch.div(energy_new, 3)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy_new.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = energy_new / scaling
        seqrow = np.arange(seq_len)
        seqrow = np.repeat(seqrow, seq_len)
        arrrow = seqrow.reshape(seq_len, seq_len)
        seqcol = np.arange(seq_len)
        arrcol = np.tile(seqcol, (seq_len, 1))
        subarr = arrrow - arrcol
        gammaarr = np.where(subarr > 0, 0, subarr)
        gammaarr = np.exp(-np.square(gammaarr) / 5)
        gammaarr = np.triu(gammaarr, k=1)
        gammaarr = gammaarr + 1
        gamma = torch.from_numpy(gammaarr).to(device)
        gamma = gamma.unsqueeze(0).unsqueeze(0)
        newatt = att * gamma
        newatt = F.softmax(newatt, dim=-1)
        att = self.att_drop(newatt)
        att = att.float()
        out = torch.einsum('bhal, bhlv -> bhav ', att, values_x).to(device)
        # out=(1000,150,10)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
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


    def __init__(self, feature_size, noise_size, batch_size, hidden_size, **kwargs):

        super(LSTMModel, self).__init__(**kwargs)

        self.feature_size = feature_size
        self.num_hiddens = self.rnn.hidden_size
        self.batch_size = batch_size
        print("self.feature_size:", self.feature_size)
        print("self.num_hiddens:", self.num_hiddens)
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

        x = self.Linear(x)
        xL = self.LayerNorm(x)
        xH = self.LayerNorm(xh)
        xT = self.LayerNorm(xT)
        output = self.LT_MultiHeadAttention(xL, xT, xH)
        output = self.dropout(output)
        x_new = x + output
        x1 = self.LayerNorm(x_new)
        x1 = self.FeedForwardBlock(x1)
        x1 = self.dropout(x1)
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
    def __init__(self, in_channels=21, patch_size=16, emb_size=100, seq_length=1024):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            nn.Linear(patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        x_encode = torch.add(x, self.positions)
        return x



class OriDis_MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        seq_len = x.shape[1]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = energy / scaling
        newatt = att
        newatt = F.softmax(newatt, dim=-1)
        att = self.att_drop(newatt)
        att = att.float()
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out=(1000,150,10)
        out = rearrange(out, "b h n d -> b n (h d)")
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
        x = self.Linear(x)
        xL = self.LayerNorm(x)
        # xH = self.LayerNorm(xh)
        # xT = self.LayerNorm(xT)
        output = self.Dis_MultiHeadAttention(xL)
        output = self.dropout(output)
        x_new = x + output
        x1 = self.LayerNorm(x_new)
        x1 = self.FeedForwardBlock(x1)
        x1 = self.dropout(x1)
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

    def forward(self, x):
        x1 = x[:, :, :2]
        xh_origin = x1+ self.pos_embed
        # 使用Enhanced Transformer提取真实轨迹与生成轨迹特征
        feature = self.Dis_TransformerEncoder(xh_origin)
        y = self.ClassificationHead(feature)
        return y

# if __name__ == '__main__':
    # x = torch.randn(1000, 150, 3)
    # # d = Discriminator()
    # y = d(x)
    # print("y:", y)
    # print(y)



