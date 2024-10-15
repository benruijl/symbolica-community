from symbolica_community import Expression, E, TensorNetwork
import symbolica_community
import random

x = Expression.parse(
    "γ(aind(loru(4,3),bis(4,2),bis(4,13)))*\
       ( P(2,aind(lord(4,11)))*γ(aind(loru(4,11),bis(4,13),bis(4,1))) + mq*id(aind(bis(4,13),bis(4,1))) )*\
       W(1,aind(bis(4,2))) * W(3,aind(lord(4,3)))")
tn = TensorNetwork(x)
tn.contract()
t = tn.result()
print(t)
params = [E("𝑖")]

params += TensorNetwork(E("W(1,aind(bis(4,2)))")).result()
params += TensorNetwork(E("W(3,aind(bis(4,2)))")).result()
params += TensorNetwork(E("P(2,aind(lord(4,2)))")).result()
constants = {E("mq"): E("173")}
e=t.evaluator(constants=constants, params=params, functions={})
c = e.compile("f","ff","fff")


e_params = [random.random()+1j*random.random() for i in range(len(params))]

print(c.evaluate_complex([e_params])[0])
