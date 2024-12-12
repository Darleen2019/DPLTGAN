import torch
from torch import nn
import numpy as np
import math
from datetime import datetime

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

if __name__ == "__main__":
    sineact = SineActivation(6, 64)
    cosact = CosineActivation(1, 64)
    X_train = np.load("../T-drive T-drive traj_data.npy", allow_pickle=True)
    X_train = X_train[:, :160, :]
    timestamp_column = X_train[:,:,1]
    timestamp_objects = [list(map(lambda ts: datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S"), ts_list)) for ts_list in
                         timestamp_column]

    years_list = []
    months_list = []
    days_list = []
    hours_list = []
    minutes_list = []
    seconds_list = []

    # Loop through Timestamps and extract components
    for ts_list in timestamp_objects:
        years_list.append([ts.year for ts in ts_list])
        months_list.append([ts.month for ts in ts_list])
        days_list.append([ts.day for ts in ts_list])
        hours_list.append([ts.hour for ts in ts_list])
        minutes_list.append([ts.minute for ts in ts_list])
        seconds_list.append([ts.second for ts in ts_list])

    # Convert lists to NumPy arrays
    years_array = np.array(years_list)
    months_array = np.array(months_list)
    days_array = np.array(days_list)
    hours_array = np.array(hours_list)
    minutes_array = np.array(minutes_list)
    seconds_array = np.array(seconds_list)

    # Stack extracted components with original data
    new_data = np.dstack((X_train, years_array, months_array, days_array, hours_array, minutes_array, seconds_array))
    # 删除原来的时间特征列
    new_data = np.delete(new_data, 1, axis=2)
    # 对现有的拆开的时间特征进行编码（时间特征分为年月日时分秒，因此是后六列数据
    t2v_encode = sineact(new_data[:,:,-6:])
    tau = torch.tensor([
        [2023, 8, 25, 12, 0, 0],
        [2023, 8, 25, 13, 30, 0],
        [2023, 8, 26, 9, 15, 0],
        # ... more timestamps ...
    ], dtype=torch.float16)
    print(sineact(tau).shape)
    # print(cosact(torch.Tensor([[7]])).shape)
    
