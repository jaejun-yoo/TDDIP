import torch 
import numpy as np
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class interKernelPrune(prune.BasePruningMethod):

    """
    Prune the kernels between feature maps. Ranks kernels based on the mean contribution its' weights have on the 
    first order taylor expansion of the loss, then remove an 'amount' proportion of these kernels from the module

    NOTE: currently pytorch prune doesn't support such pruning, so it can currently only be applied once during training. 
    Otherwise we need to implement a new pruningContainer. 
    """

    PRUNING_TYPE = 'structured'

    def __init__(self, amount=0.1):
        # Check range of validity of pruning amount
        if (type(amount) != float):
            raise Exception 
        assert((amount > 0.0) & (amount < 1.0))
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # the pytorch interface doesn't support intrakernel structured pruning:
        # would need to implement new pruning container

        mask = default_mask.clone()

        weight = t.flatten(start_dim=2, end_dim=3) 
        grad = t.grad.flatten(start_dim=2, end_dim=3)
        product = torch.abs((weight*grad).mean(dim=2))

        treshold = np.percentile(product.detach().cpu().numpy(), self.amount*100)
        mask[product < treshold, :, :] = 0 

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super(interKernelPrune, cls).apply(module, name, amount=amount)

def interkernel_structured(module, name, amount):
    interKernelPrune.apply(module, name, amount)
    return module

def pruneOnStep(opt, step, model):

    """
    prune the convolutional layers in 'model' by a proportion specified in 'opt.amount', if step is one of the pruning steps.
    """ 

    if step in opt.pruneSteps:

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if opt.method == 'interkernel':
                    interkernel_structured(module, name="weight", amount=opt.amount)
                else:   
                    prune.ln_structured(module, name="weight", amount=opt.amount, n=1, dim=1)

        sparsities = [] 
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                sparsities += [(100. * float(torch.sum(module.weight == 0))/ float(module.weight.nelement()))] 

        print("PRUNING STEP - sparsity in conv. layers (%) : " , sum(sparsities)/len(sparsities))

def removeReparam(model):

    """ 
    Remove the reparametrization from pruning - necessary before saving model
    """ 

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")




        
