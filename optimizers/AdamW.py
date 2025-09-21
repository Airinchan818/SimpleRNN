import numpy as np 

class AdamW :
    def __init__ (self,lr = 1e-3,beta1 = 0.9,beta2 = 0.999, weight_decay = 4e-3,clipnorm=None):
        self.lr = lr 
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.model_weight = None 
        self.model_grad = None 
        self.Momentum = None 
        self.RMS = None 
        self.epoch = 0 
        self.clipnorm = clipnorm
    
    def build_component (self) :
        if isinstance(self.model_weight,list) :
            self.Momentum = list()
            self.RMS = list()
            for weight in self.model_weight :
                self.Momentum.append(np.zeros_like(weight))
                self.RMS.append(np.zeros_like(weight))
    
    def apply_weight(self,weight : list) :
        self.model_weight = weight
    
    def apply_grad (self,grad : list) :
        self.model_grad = grad 
    
    def __execution (self,w,g,m,r,epoch) :
        if self.clipnorm is not None :
            norm = np.linalg.norm(g)
            if norm > self.clipnorm :
                scale = self.clipnorm / norm 
                g = scale * g 
        moment = self.beta1 * m + (1 - self.beta1) * g 
        rms = self.beta2 * r + (1 - self.beta2) * np.power(g,2)
        moment_hat = moment / ( 1 - np.power(self.beta1,epoch + 1))
        rms_hat = rms / (1 - np.power(self.beta2,epoch + 1))
        w_decay = (1 - self.lr * self.weight_decay)
        weight = w * w_decay - self.lr * moment_hat / np.sqrt(rms_hat + 1e-6 )
        return weight,moment,rms

    def step(self) :
        if self.Momentum is None or self.RMS is None :
            self.build_component()
        for i in range(len(self.model_weight)) :
            component = self.__execution(self.model_weight[i],
                                         self.model_grad[i],
                                         self.Momentum[i],
                                         self.RMS[i],self.epoch)
            self.model_weight[i] = component[0]
            self.Momentum[i] = component[1]
            self.RMS[i] = component[2]
        
        self.epoch += 1 
        return self.model_weight