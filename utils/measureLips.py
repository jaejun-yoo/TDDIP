import torch
from scipy.linalg import svdvals

def spectralNorms(seq):

    """
    given sequential model, compute spectral norm of the reshaped module weights. 
    """

    norms = []

    for layer in seq.children():
        if isinstance(layer, torch.nn.Conv2d):
            w = layer.weight.detach()
            s = w.size()
            w_ = w.reshape(s[1], s[0]*s[2]*s[3]).cpu().numpy() 
            norm = svdvals(w_).max()
            norms.append(norm)
        elif isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            norm = svdvals(w).max()
            norms.append(norm)
    return norms