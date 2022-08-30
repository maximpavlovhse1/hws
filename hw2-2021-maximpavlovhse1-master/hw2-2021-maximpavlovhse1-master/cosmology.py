# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:34:22 2021

@author: I
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import opt


file = open('jla_mub.txt', 'r')
data = np.genfromtxt(file, delimiter=' ', skip_header=1)

z = data[:, 0]
mu = data[:, 1]

res = opt.gauss_newton(mu, opt.f, opt.j, (z, 50, 0.5))
res2 = opt.lm(mu, opt.f, opt.j, (z, 50, 0.5))
print(res2)



plt.scatter(z, mu)

plt.plot(z , opt.f(*res.x), color = 'pink', label = "gauss newton")
plt.plot(z, opt.f(*res2.x), color = 'orange', label = "lm")
plt.legend(title = "на самом деле тут две прямые" + "\n" + "по параметрам из двух различных методов", ncol = 2)
plt.xlabel("Z")
plt.ylabel("MU")
plt.savefig("mu-z.png")
plt.figure()




d = {"Gauss-Newton": {"H0": float(np.around(res.x[1])), "Omega": float(np.around(res.x[2], 1)) , "nfev": float(res.nfev) }, 
"Levenberg-Marquardt": {"H0": float(np.around(res2.x[1])), "Omega": float(np.around(res2.x[2], 1)), "nfev": float(res2.nfev)}
                     }


with open("parameters.json", "w") as f: 
    json.dump(d, f)

 




plt.plot(np.arange(res2.nfev), res2.cost, color = "blue", label = "lm")
plt.plot(np.arange(res.nfev), res.cost, color = "red", label = "gauss newton")
plt.xlabel('Итерационный шаг')
plt.ylabel('sum(0.5 * (mu-f)^2)')
plt.legend()
plt.savefig("cost.png")
plt.figure()