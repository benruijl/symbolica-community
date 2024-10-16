from symbolica_community import Expression, S,E, TensorNetwork
import symbolica_community
import random


mu = E("mink(4,mu)")
nu = E("mink(4,nu)")
i = E("bis(4,i)")
j = E("bis(4,j)")
k = E("bis(4,k)")
gamma,p,w,mq,id = S("γ","P","W","mq","id")
x = gamma(mu,i,k)*(p(2,nu)*gamma(nu,k,j)+mq*id(k,j))*w(1,i)*w(3,mu)

tn = TensorNetwork(x)
tn.contract()
t = tn.result()
print(t)
params = [E("𝑖")]

params += TensorNetwork(E("W(1,bis(4,2))")).result()
params += TensorNetwork(E("W(3,bis(4,2))")).result()
params += TensorNetwork(E("P(2,mink(4,2))")).result()
constants = {E("mq"): E("173")}
e=t.evaluator(constants=constants, params=params, functions={})
c = e.compile("f","ff","fff")


e_params = [random.random()+1j*random.random() for i in range(len(params))]

print(c.evaluate_complex([e_params])[0])
