import numpy as np

# baseline methods

def ITLinQ(H_raw, Pmax, noise_var, PFs):
    H = H_raw * Pmax / noise_var
    n = np.shape(H)[0]
    prity = np.argsort(PFs)[-1:-n-1:-1]
    flags = np.zeros(n)
    M = 10 ** 2.5
    eta = 0.5
    flags[prity[0]] = 1
    for pair in prity[1:]:
        SNR = H[pair,pair]
        INRs_in = [H[TP,pair] for TP in range(n) if flags[TP]]
        INRs_out = [H[pair,UE] for UE in range(n) if flags[UE]]
        max_INR_in = max(INRs_in)
        max_INR_out = max(INRs_out)
        if max(max_INR_in,max_INR_out) <= M * (SNR ** eta):
            flags[pair] = 1
    return flags * Pmax

def wmmse(H, Pmax, noise_var):
    h2 = np.copy(H)
    h = np.sqrt(h2)
    m = H.shape[1]
    N = H.shape[0]
    v = np.ones((N,m))*np.sqrt(Pmax)/2
    T = 100
    v2 = np.expand_dims(v ** 2, axis=2)

    u = (np.diagonal(h,axis1=1,axis2=2) * v) / (np.matmul(h2,v2)[:,:,0] + noise_var)
    w = 1 /(1 - u *np.diagonal(h,axis1=1,axis2=2) *v)
    N = 1000
    for n in np.arange(T):

        u2 = np.expand_dims(u**2, axis=2)
        w2 = np.expand_dims(w, axis=2)
        v = (w *u *np.diagonal(h,axis1=1,axis2=2)) / (np.matmul(np.transpose(h2,(0,2,1)),(w2*u2)))[:,:,0]
        v = np.minimum(np.sqrt(Pmax),np.maximum(0,v))
        v2 = np.expand_dims(v**2,axis=2)
        u = (np.diagonal(h,axis1=1,axis2=2)*v)/ ( np.matmul(h2,v2)[:,:,0] + noise_var)
        w = 1 /(1 - u*np.diagonal(h,axis1=1,axis2=2)*v)
    p = v**2
    return p
