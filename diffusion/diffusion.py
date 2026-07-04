import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

#bid 13.1 - ask 17.4, for S0 = 52.93
S0 = 52.93
L = 1001
sigma = 0.1586
T = 17/365
r = 0.01
X = 37.5
S_min, S_max = 0.5*X, 10*X
# T = tau*m
tau = 0.0001

m = int(T/tau)

# tau <= Delta^2/2*D <- stability condition
D = 1/2 * sigma**2

# at tau = 0 we have no drift term
x_min = np.log(1*S_min)
x_max = np.log(1*S_max)
x = np.linspace(x_min, x_max, L)
delta = (x_max-x_min)/(L-1)
print(delta)

print('Checking Stability condition')
if not ((tau <= delta**2 /(2*D))):
    print('Delta=', delta)
    print('Delta^2=', delta**2)
    print('D=', D)
    print('tau=', tau)
    print(tau, '<=', delta**2/(2*D))
else:
    print('Stability Condition is true')

# Initial conditions:

u = np.zeros(L)
#gamma = 2*r/sigma**2

def boundary(x):
    return max(np.exp(x) - X, 0.0)

for i in range(L):
    u[i] = max(boundary(x[i]), 0)
u[0] = 0

alpha = -D / delta**2
beta = alpha * tau
print('beta', beta)

matrixA = 0.5 * np.array(([ 1 + np.exp(beta), 1 - np.exp(beta)], 
                          [1 - np.exp(beta), 1+ np.exp(beta)]))

matrixB = 0.5 * np.array(([1 + np.exp(2*beta), 1 - np.exp(2*beta)], 
                          [1 - np.exp(2*beta), 1 + np.exp(2*beta)]))


def multiplyA(vec):
    out = vec.copy()
    out[-1] *= np.exp(beta)
    for i in range(int(L/2)):
        out[i*2:i*2 + 2] = np.matmul(matrixA, out[i*2:i*2+2])
    return out


def multiplyB(vec):
    out = vec.copy()
    out[0] *= np.exp(2*beta)
    for i in range(int(L/2)):
        out[i*2+1:i*2+3] = np.matmul(matrixB, out[i*2+1:i*2+3])
    return out


for j in range(m):
    u_new = multiplyA(multiplyB(multiplyA(u)))
    tau_now = (j + 1) * tau
    u_new[0] = 0.0
    u_new[-1] = np.exp(x[-1] + 0.5 * sigma**2 * tau_now) - X
    u = u_new

C = np.exp(-r*tau*m)*u

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

def bs_call(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


# x = log(S) + (r-1/2*sigma**2)*tau
# => S(0) = e^{x-(r-1/2*sigma**2)*m*tau)
S = np.exp(x-(r-1/2*sigma**2)*m*tau)
analytical_price = bs_call(52.93, 37.5, r,T,sigma)
plt.figure(figsize=(12,8))
plt.title('Black Scholes as Diffusion Equation', fontsize=14)
plt.plot(S,C)
plt.xlabel(rf'S=$exp(x-(r-0.5*sigma^2))*m*tau)$', fontsize=14)
plt.ylabel(rf'C(S,t=0) = $exp(-r*\tau *m) u(\tau * m)$', fontsize=14)
idx = int(np.argmin(np.abs(S - S0)))
print(f"Closest index to S0: {idx}, S={S[idx]}, C={C[idx]}")
plt.vlines(S0, C[0], C[-1], linestyle='--', label=rf'$S_0$', color='red')
plt.hlines(C[idx], S[0], S[-1], linestyle='--', color='blue', label=rf'C(S≈S0)={C[idx]:.4f}')
plt.scatter(S[0],C[0],alpha=0.0,label=rf'analytical:{analytical_price:.4f}')
plt.grid()
plt.legend(prop={'size': 16})
plt.savefig('diffusion.png')
plt.show()