import numpy as np 

class layers :
    def __init__ (self) :
        pass 

    def get_weight(self) :
        raise NotImplementedError
    
    def get_gradient (self) :
        raise NotImplementedError
    
    def __call__(self,x) :
        raise NotImplementedError
    
    def backward(self,grad_out) :
        raise NotImplementedError
    
    def update_weight(self,w ) :
        raise NotImplementedError

class Embedding (layers):
    def __init__(self,units,outputs):
        super().__init__()

        self.units = units 
        self.outputs = outputs 
        self.weight = None 
        self.gradient = None 
        self.hist = None 
    
    def build_weight(self) :
        self.weight = np.random.rand(self.units,self.outputs)
    
    def __call__(self,x):
        if self.weight is None :
            self.build_weight()
        self.hist = x 
        return self.weight[x]

    def backward(self,grad_out) :
        self.gradient = np.zeros_like(self.weight)
        for i,token_ide in enumerate(self.hist.reshape(-1)):
            self.gradient[token_ide]+= grad_out.reshape(-1,self.weight.shape[1])[i]
    
    def get_weight(self):
        return [self.weight]

    def get_gradient(self):
        return [self.gradient]
    
    def update_weight(self, w):
        self.weight = w 
    

class Linear (layers) :
    def __init__ (self,units) :
        self.units = units 
        self.weight = None 
        self.bias = None 
        self.xhist = None 
        self.gradient_Weight = None 
        self.gradient_bias = None 
    
    def build_weight(self,features) :
        glorot_uniform = np.sqrt(6 / (features + self.units))
        self.weight = np.random.uniform(
            low=-glorot_uniform,high=glorot_uniform,
            size=(features,self.units)
        )
        self.bias = np.zeros((1,self.units))
    
    def __call__(self, x):
        if self.weight is None or self.bias is None :
            self.build_weight(x.shape[-1])
        self.xhist = x 
        return np.matmul(x,self.weight) + self.bias 
    
    def backward(self, grad_out):
        grad = np.matmul(np.swapaxes(self.xhist,-2,-1),grad_out)
        self.gradient_Weight = grad 
        self.gradient_bias = np.sum(grad_out,axis=0,keepdims=True)
        return np.matmul(grad_out,self.weight.swapaxes(-2,-1))
    
    def get_gradient(self):
        return [self.gradient_Weight,self.gradient_bias]
    
    def get_weight(self):
        return [self.weight,self.bias]
    
    def update_weight(self, w):
        self.weight = w 
    
    def update_bias (self,b) :
        self.bias = b 
    

class SimpleRNN (layers) :
    def __init__(self,units,return_sequence = False) :
        super().__init__() 
        self.units = units 
        self.return_sequence = return_sequence
        self.weight_s = None 
        self.weight_h = None 
        self.bias = None 
        self.grad_ws = None 
        self.grad_wh = None 
        self.grad_bias = None 
        self.x_hits = list()
        self.hidden_state = None
        self.h_prevhist = list()
        self.hist_tanh = list()
        self.batch_size = None 
        self.sequence_size = None 
    
    def build_weight (self,features) :
        he_uniform = np.sqrt(6 / features)
        self.weight_s = np.random.uniform(
            low=-he_uniform,high=he_uniform,
            size=(features,self.units)
        )
        self.weight_h = np.random.uniform(
            low = -he_uniform,high=he_uniform,
            size=(self.units,self.units)
        )
        self.bias = np.zeros((1, self.units))
    
    def tanh_(self,x) :
        x = np.tanh(x)
        self.hist_tanh.append(x)
        return x 
    
    def d_tanh(self,dth,grad) :
        return (1 - dth**2) * grad 
    
    def execution (self,x,h_t) :
        sequence_logits = np.matmul(x,self.weight_s)
        hidden_logits = np.matmul(h_t,self.weight_h)
        logits = (sequence_logits + hidden_logits) + self.bias 
        return self.tanh_(logits)

    def __call__ (self,x) :
        if self.weight_s is None or self.weight_h is None :
            self.build_weight(x.shape[-1])
            self.hidden_state = np.zeros((x.shape[0],self.units))
            self.h_prevhist.append(self.hidden_state)
        self.batch_size = x.shape[0]
        self.sequence_size = x.shape[1]
        if x.ndim <= 2:
            print(x.shape)
        seq_out = list()
        for t in range(x.shape[1]) :
            xt = x[:,t,:]
            self.x_hits.append(xt)
            self.hidden_state = self.execution(xt,self.hidden_state)
            self.h_prevhist.append(self.hidden_state)
            seq_out.append(self.hidden_state)
        
        seq_out = np.stack(seq_out,axis=1)
        
        if self.return_sequence is True :
            return seq_out
        else :
            return seq_out[:,-1,:]
    
    def backward(self, grad_out):
        d_xhist = np.zeros_like(self.weight_s)
        d_hiddst = np.zeros_like(self.weight_h)
        d_b = np.zeros_like(self.bias)
        d_next = np.zeros((self.batch_size,self.units))
        d_x_next = list() 
        for t in reversed(range(self.sequence_size)) :
            dh = grad_out[:,t,:] + d_next
            h = self.hist_tanh[t]
            h_p = self.h_prevhist[t]
            xt = self.x_hits[t]

            dtanh = self.d_tanh(h,dh)
            d_xhist += np.dot(xt.T,dtanh)
            d_hiddst += np.dot(h_p.T,dtanh)
            d_b += np.sum(dtanh,axis=0,keepdims=True)
            dx = np.dot(dtanh,self.weight_s.T)
            d_x_next.insert(0,dx)
            d_next += np.matmul(dtanh,self.weight_h.T)
        
        self.grad_ws = d_xhist
        self.grad_wh = d_hiddst
        self.grad_bias = d_b
        self.hist_tanh.clear()
        self.x_hits.clear()
        self.h_prevhist.clear()
        self.hist_tanh.clear()

        
        return np.stack(d_x_next,axis=1)
    
    def get_weight(self):
        return [self.weight_s,self.weight_h,self.bias]
    
    def get_gradient(self):
        return [self.grad_ws,self.grad_wh,self.grad_bias]
    
    def update_weight(self, ws):
        self.weight_s = ws
    
    def update_weight_h (self,wh) :
        self.weight_h = wh 
    
    def update_bias (self,b) :
        self.bias = b 


class ReLU(layers) :
    def __init__(self):
        super().__init__()
        self.hist = None 
    
    def __call__(self, x):
        self.hist = x 
        return np.maximum(0,x)
    
    def backward(self, grad_out):
        return np.where(self.hist > 0,1,0) * grad_out
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 

class Sigmoid (layers) :
    def __init__(self):
        super().__init__()
        self.hist = None 
    
    def __call__(self, x):
        self.hist = 1 / (1 + np.exp(-x))
        return self.hist
    
    def backward(self, grad_out):
        grad =  self.hist * (1 - self.hist)
        return grad * grad_out 
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 

class BinaryCrossEntropy () :
    def __init__(self):
        self.epsilon = 1e-6
        self.cache_y_true = None 
        self.cache_y_pred = None 
    
    def __call__(self,y_true,y_pred) :
        self.cache_y_true = y_true
        self.cache_y_pred = y_pred 
        n = len(y_true)
        loss = (-1/n) * np.sum(y_true * np.log(y_pred + self.epsilon) + 
                               (1 - y_true) * np.log(1 - y_pred  + self.epsilon))
        return loss 
    
    def backward(self) :
        return (self.cache_y_pred - self.cache_y_true) / len(self.cache_y_true)


class Sequential (layers) :
    def __init__ (self,component : list) :
        self.component = component 
    
    def __call__(self,x) :
        for component in self.component :
            x = component(x)
        return x 
    
    def backward(self, grad_out):
        for component in reversed(self.component) :
            if isinstance(component,layers) :
                grad_out = component.backward(grad_out)
    
    def get_weight(self):
        weight = list()
        for component in self.component :
            if isinstance(component,layers) :
                wg = component.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
        return weight
    
    def get_gradient(self):
        gradient = list()
        for component in self.component :
            if isinstance(component,layers) :
                gl = component.get_gradient()
                if gl is not None :
                    for g in gl :
                        gradient.append(g)
        return gradient
    
    def update_weight(self, w : list):
        counter = 0 
        for component in self.component :
            if isinstance(component,Linear) :
                component.update_weight(w[counter])
                counter += 1 
                component.update_bias(w[counter])
                counter +=1 
            
            if isinstance(component,Embedding) :
                component.update_weight(w[counter])
                counter +=1 
            
            if isinstance(component,SimpleRNN) :
                component.update_weight(w[counter])
                counter +=1
                component.update_weight_h(w[counter])
                counter +=1 
                component.update_bias(w[counter])
                counter +=1 

class GlobalAveragePooling (layers) :
    def __init__(self,axis):
        super().__init__()
        self.x_hist = None 
        self.axis = axis 
    
    def __call__(self,x) :
        self.x_hist = x 
        return np.mean(x,axis=self.axis,keepdims=False )
    
    def backward(self, grad_out):
        if isinstance(self.x_hist,np.ndarray):
            if self.axis is None :
                N = self.x_hist.size
            elif isinstance(self.axis,int) :
                N = self.x_hist[self.axis]
            else :
                N = 1 
                for ax in self.axis :
                    N *= self.x_hist.shape[ax]
        grad = np.expand_dims(grad_out,axis = self.axis)
        grad = np.broadcast_to(grad/N,self.x_hist.shape)
        return grad 
    
    def get_gradient(self):
        return None 
    
    def get_weight(self):
        return None 
    
    def update_weight(self, w):
        pass 
            