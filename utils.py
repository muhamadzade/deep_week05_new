import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    x = x.cuda() if torch.cuda.is_available() else x
    return Variable(x, volatile=volatile)


def detach(x):
    """ Detach hidden states from their history."""
    #print(type(x),Variable)
    
    if type(x) == Variable:
        return Variable(x.data)
    else:
        
#        print(type(x),Variable)
#        if (x.size()):
#            print ("00000000000000")
        temp=[]
        #print(x)
        #print()
        if isinstance(x,tuple):
            return
        elif isinstance(x,torch.Tensor) and bool(x.size())==False :
            return
            
        for v in x:
           temp.append(detach(v))
        temp=tuple(temp)
        return temp    
    
    #return Variable(x.data[0]) if type(x) == Variable else tuple(detach(v) for v in x)