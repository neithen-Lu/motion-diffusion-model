import torch

def temporal_consistency_metric(loader):
    tcm = 0
    # print('output',loader.batches[0]["output"].shape)
    # B,H,W,L = iterator_generated[0].shape
    for i in range(len(loader.batches)):
        data = loader.batches[0]["output"]
        diff = data[:,:,:,1:] - data[:,:,:,:-1]
        a = torch.norm(diff, p='fro', dim=(1, 2))
        tcm += torch.mean(a)
    tcm = tcm / len(loader.batches)
    return float(tcm)
    