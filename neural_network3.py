class add_node():
    def __init__(self,x,y):
        self.res=x+y
        self.grad_x=1
        self.grad_y=1
class multiply_node():
    def __init__(self,x,y):
        self.res=x*y
        self.grad_x=1
        self.grad_y=1
class square_node():
    def __init__(self,x):
        self.res=pow(x,2)
        self.grad_x=2*x
class cell():#神经元
    def __init__(self,x,w):
        self.grad=[]
        self.res=0
        for i in range(len(x)):
            self.res+=x[i]*w[i]
            self.grad.append(x[i])
            self.res+=w[len(x)]
            self.grad.append(1)
x=[[0.1,0.8],[0.8,0.2]]
y=[1,0]
w=[0.1,0.2,0.3]
for i in range(100000):
    out0=cell(x[0],w)#神经元0
    out1=cell(x[1],w)#神经元1
    print('out',[out0.res,out1.res])
    d0=add_node(out0.res,-y[0])
    pn0=square_node(d0.res)
    d1 = add_node(out1.res, -y[1])
    pn1 = square_node(d1.res)
    an1=add_node(pn0.res,pn1.res)
    print('loss',an1.res)
    td0=an1.grad_x*pn0.grad_x*d0.grad_x*out0.grad[0]+an1.grad_y*pn1.grad_x*d1.grad_x*out1.grad[0]
    td1 = an1.grad_x * pn0.grad_x * d0.grad_x*out0.grad[1] + an1.grad_y * pn1.grad_x * d1.grad_x * out1.grad[1]
    td2 = an1.grad_x * pn0.grad_x * d0.grad_x*out0.grad[2] + an1.grad_y * pn1.grad_x * d1.grad_x * out1.grad[2]
    w[0]-=td0*0.01
    w[1]-=td1*0.01
    w[2]-=td2*0.01