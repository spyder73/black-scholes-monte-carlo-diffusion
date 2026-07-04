import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
#Asian call options data from here: https://www.diva-portal.org/smash/get/diva2:301070/FULLTEXT01.pdf
#Lookback options data from here: https://www.researchgate.net/publication/351788645_Estimating_Lookback_Price_Using_Monte_Carlo_Simulation_and_Binomial_Lattice/fulltext/60abfbcd45851522bc15145a/Estimating-Lookback-Price-Using-Monte-Carlo-Simulation-and-Binomial-Lattice.pdf
#Lookback options analytical formula from here: https://www.quantstart.com/articles/Floating-Strike-Lookback-Option-Pricing-with-C-via-Analytic-Formulae/

PARAMS_CSV = "params.csv"
CUT = 0.2

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

def bs_call(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

def binary_call(S0, K, r, T, sigma):
    d_b = (np.log(S0/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    BS_b = np.exp(-r*T) * norm_cdf(d_b - sigma * np.sqrt(T))
    return BS_b

def binary_put(S0, K, r, T, sigma):
    d_2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return np.exp(-r*T)*norm_cdf(-d_2)

def bs_put(S, X, r, T, sigma):
    d = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    N_1 = norm_cdf(-d + sigma*np.sqrt(T))
    N_2 = norm_cdf(-d)
    
    p = X * np.exp(-r*(T)) * N_1 - S * N_2
    return p

def asian_call(S0,K,r,T,sigma, N):
    sigma_z = sigma * np.sqrt((2*N+1)/(6*(N+1)))
    rho = ((r-sigma**2/2) + sigma_z**2)/2
    factor = np.exp((rho-r)*T)
    d1 = (np.log(S0 / K) + (rho + 0.5*sigma_z**2) * T) / (sigma_z * np.sqrt(T))
    d2 = (np.log(S0 / K) + (rho - 0.5*sigma_z**2) * T) / (sigma_z * np.sqrt(T))
    return factor*(S0 * norm_cdf(d1) - K * np.exp(-rho*T)*norm_cdf(d2))

def asian_put(S0,K,r,T,sigma,N):
    sigma_z = sigma * np.sqrt((2*N+1)/(6*(N+1)))
    rho = ((r-sigma**2/2) + sigma_z**2)/2
    factor = np.exp((rho-r)*T)
    d1 = (np.log(S0 / K) + (rho + 0.5*sigma_z**2) * T) / (sigma_z * np.sqrt(T))
    d2 = (np.log(S0 / K) + (rho - 0.5*sigma_z**2) * T) / (sigma_z * np.sqrt(T))
    return factor*(K*np.exp(-rho*T)*norm_cdf(-d2) - S0*norm_cdf(-d1))

# Functions for Lookback:
def a1(S,H,r,sigma,T):
    return (np.log(S/H) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
def a2(S,H,r,sigma,T):
    return a1(S,H,r,sigma,T) - sigma*np.sqrt(T)
def a3(S,H,r,sigma,T):
    return a1(S,H,r,sigma,T) - 2*r*np.sqrt(T)/sigma

def lookback_call(S0,r,T,sigma,factor=0.1):
    # estimating min price of S during period T
    m = (1-factor)*S0
    p1 = S0*norm_cdf(a1(S0,m,r,sigma,T))
    p2 = -m*np.exp(-r*T)*norm_cdf(a2(S0,m,r,sigma,T))
    p3 = -S0*sigma**2 /(2*r) * (norm_cdf(-a1(S0,m,r,sigma,T)) - np.exp(-r*T)*(m/S0)**(2*r/sigma**2) * norm_cdf(-a3(S0,m,r,sigma,T)))
    return p1 + p2 + p3

def lookback_put(S0,r,T,sigma,factor=0.1):
    # estimating max price of S during period T
    M = (1+factor)*S0
    p1 = -S0*norm_cdf(-a1(S0,M,r,sigma,T))
    p2 = M*np.exp(-r*T)*norm_cdf(-a2(S0,M,r,sigma,T))
    p3 = S0*sigma**2/(2*r) * (norm_cdf(a1(S0,M,r,sigma,T)) - np.exp(-r*T)*(M/S0)**(2*r/sigma**2)*norm_cdf(a3(S0,M,r,sigma,T)))
    return p1 + p2 + p3

# Load params
p = pd.read_csv(PARAMS_CSV).iloc[0]
label = p["label"]
option_type = p["option_type"]
S0, K, r, sigma, T, N = p["S0"], p["K"], p["r"], p["sigma"], p["T"]/365.0, p["stepSize"]
bid = p.get("bid")
ask = p.get("ask")

# Load MC results
df = pd.read_csv(f"results/discounted_{label}.csv")
running_mean = df["discounted"].expanding().mean()

res = pd.read_csv(f"results/result_{label}.csv").iloc[0]
price = res["price"]
stderr = res["stderr"]

plt.figure(figsize=(12, 8))
size = len(running_mean)
plt.plot(df["i"][int(CUT*size):], running_mean[int(CUT*size):], label="MC Running mean")
# MC result + std
plt.axhline(price, color='yellow',label=f"MC Price = {price:.4f}")
plt.axhspan(price - stderr, price + stderr, alpha = 0.5, color = "red", label=rf"$\sigma$ = {stderr}")

sigma_distance = None

if option_type == "european_call":
    analytic = bs_call(S0, K, r, T, sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
    
if option_type == "european_put":
    analytic = bs_put(S0, K, r, T, sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr

if option_type == "asian_call":
    analytic = asian_call(S0,K,r,T,sigma,N)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
        
if option_type == "asian_put":
    analytic = asian_put(S0,K,r,T,sigma,N)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
        
if option_type == "binary_call":
    analytic = binary_call(S0,K,r,T,sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
        
if option_type == "binary_put":
    analytic = binary_put(S0,K,r,T,sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
            
if option_type == "lookback_call":
    analytic = lookback_call(S0,r,T,sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
        
if option_type == "lookback_put":
    analytic = lookback_put(S0,r,T,sigma)
    plt.axhline(analytic, color="red", linestyle="--", label=f"Analytical = {analytic:.4f}")
    if stderr > 0: 
        sigma_distance = (price - analytic) / stderr
        
if pd.notna(bid) and pd.notna(ask):
    if option_type == "lookback_call" or option_type == "lookback_put":
        plt.axhspan(bid, ask, alpha=0.3, color="green", label=f"Bid/Ask [{bid:.2f}, {ask:.2f}] from other MC")
    else:
        plt.axhspan(bid, ask, alpha=0.3, color="green", label=f"Bid/Ask [{bid:.2f}, {ask:.2f}]")

if sigma_distance is not None:
    plt.scatter(df["i"][-2:-1], running_mean[-2:-1], alpha=0, label = rf"Deviation $\approx$ {abs(sigma_distance):.2f}$\sigma$")
    
plt.xlabel("Number of trajectories", fontsize=14)
plt.ylabel("Option price", fontsize=14)
plt.title(label, fontsize=14)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 16})
plt.tight_layout()
plt.savefig(f'plots/{label}.png')
plt.show()