import torch

def temporal_consistency_metric(iterator_generated):
    tcm = 0
    # B,H,W,L = iterator_generated[0].shape
    for data in iterator_generated:
        diff = data[:,:,:,1:] - data[:,:,:,:-1]
        a = torch.norm(diff, p='fro', dim=(1, 2))
        tcm += torch.mean(a)
    tcm = tcm / len(iterator_generated)
    return float(tcm)
    