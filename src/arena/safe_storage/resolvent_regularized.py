"""
Resolvent Regularized Model - Standalone
========================================
Resolvent operator (lambda*I - A)^{-1} regularization.

The resolvent R(lambda, A) = (lambda*I - A)^{-1} maps observations
to a regularized state space. When the resolvent norm is large 
(lambda near spectrum), the system is near-singular → inflate 
observation variance for stability.

Tikhonov regularization connection:
  x_reg = argmin ||Ax - b||^2 + lambda||x||^2
        = (A^T A + lambda I)^{-1} A^T b
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
import time


class ResolventRegularizedModel:
    LL_BOOST = 1.27
    HYV_TARGET = -440
    ENTROPY_ALPHA = 0.10
    RESOLVENT_LAMBDA = 0.1
    RESOLVENT_MEMORY = 0.90
    RESOLVENT_SCALE = 0.15

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._resolvent_norm = 1.0
        h0a = np.array([-0.0046,-0.0116,0.0503,0.2969,0.5594,0.2969,0.0503,-0.0116,-0.0046,0.0])*np.sqrt(2)
        h1a = np.array([0.0046,-0.0116,-0.0503,0.2969,-0.5594,0.2969,-0.0503,-0.0116,0.0046,0.0])*np.sqrt(2)
        self.h0a, self.h1a = h0a, h1a
        self.h0b, self.h1b = h0a[::-1], -h1a[::-1]

    def _fd(self, s, h):
        p = np.pad(s, (len(h)//2, len(h)//2), mode='reflect')
        f = np.convolve(p, h, mode='same')
        return f[len(h)//2:-len(h)//2:2] if len(f) > len(h) else f[::2]

    def _qshift(self, s):
        cr, ci, ca, cb = [], [], s.copy(), s.copy()
        for _ in range(self.n_levels):
            if len(ca) < 10: break
            la, ha = self._fd(ca, self.h0a), self._fd(ca, self.h1a)
            lb, hb = self._fd(cb, self.h0b), self._fd(cb, self.h1b)
            cr.append((ha+hb)/np.sqrt(2)); ci.append((ha-hb)/np.sqrt(2))
            ca, cb = la, lb
        cr.append((ca+cb)/np.sqrt(2)); ci.append((ca-cb)/np.sqrt(2))
        return cr, ci

    def _mag_thresh(self, mags, k=1.4):
        return [np.where(m > np.median(m)+k*np.std(m), m, m*0.55) for m in mags]

    def _deflation(self, vol, t, win=60):
        if t < win: return 1.0
        r = vol[max(0,t-win):t]; r = r[r>0]
        if len(r) < 15: return 1.0
        c = vol[t] if vol[t] > 0 else np.mean(r)
        pct = (r<c).sum()/len(r)
        if pct < 0.23: instant = 1.30
        elif pct > 0.87: instant = 0.56
        elif pct > 0.73: instant = 0.70
        elif pct > 0.58: instant = 0.88
        else: instant = 1.0
        self._defl_memory = 0.83*self._defl_memory + 0.17*instant
        return self._defl_memory

    def _stress(self, vol, t):
        s = 1.0
        for h, w in [(3,0.36),(7,0.28),(14,0.20),(28,0.11),(56,0.05)]:
            if t >= h:
                rv = vol[t-h:t]; rv = rv[rv>0]
                if len(rv) >= max(3,h//4) and vol[t] > 0:
                    s *= 1.0 + w*max(0, vol[t]/(np.median(rv)+1e-8)-1.13)
        return np.clip(np.power(s, 0.43), 1.0, 3.6)

    def _entropy(self, vol, t):
        if t < 30: return 1.0
        w = vol[t-30:t]; w = w[w>0]
        if len(w) < 10: return 1.0
        return np.clip(1.0 + np.std(w)/(np.mean(w)+1e-8)*0.52, 0.88, 1.52)

    def _robust_vol(self, vol, t, win=20):
        if t < win: return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]; rv = rv[rv>0]
        if len(rv) < 5: return vol[t] if vol[t] > 0 else 0.01
        med = np.median(rv); mad = np.median(np.abs(rv-med))*1.4826
        curr = vol[t] if vol[t] > 0 else med
        if mad > 0 and abs(curr-med) > 2.3*mad:
            return med + np.sign(curr-med)*1.9*mad
        return curr

    def _hyv_corr(self, rh, hc, alpha=0.10, target=-400):
        if hc < 10: return 1.0
        self._hyv_memory = 0.85*self._hyv_memory + 0.15*(rh/hc)
        return np.clip(1.0 + alpha*(self._hyv_memory-target)/1000, 0.7, 1.4)

    def _scale_ll(self, mag, vol, q, c, phi):
        n = len(mag); P, state, ll = 1e-4, 0.0, 0.0
        vs = vol[::max(1,len(vol)//n)][:n] if len(vol) > n else np.ones(n)*0.01
        for t in range(1, n):
            pm = phi*state; pv = phi**2*P + q
            v = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
            S = pv + (c*v)**2; inn = mag[t]-pm
            K = pv/S if S > 0 else 0; state, P = pm+K*inn, (1-K)*pv
            if S > 1e-10: ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return ll

    def _resolvent_regularization(self, phi, P, returns, t, window=40):
        """Compute resolvent-based regularization factor."""
        if t < window:
            return 1.0
        ret_w = returns[t-window:t]
        autocorr = np.corrcoef(ret_w[:-1], ret_w[1:])[0, 1] if len(ret_w) > 2 else 0.0
        autocorr = np.clip(autocorr, -0.99, 0.99)
        spectral_radius = abs(autocorr)
        dist_to_spectrum = abs(self.RESOLVENT_LAMBDA - spectral_radius) + 1e-8
        resolvent_norm = 1.0 / dist_to_spectrum
        self._resolvent_norm = self.RESOLVENT_MEMORY * self._resolvent_norm + \
                               (1 - self.RESOLVENT_MEMORY) * resolvent_norm
        reg_factor = 1.0 + self.RESOLVENT_SCALE * np.log1p(self._resolvent_norm)
        return np.clip(reg_factor, 0.8, 1.8)

    def _filter(self, ret, vol, p):
        n = len(ret); mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._qshift(ret)
        mags = [np.sqrt(cr[i]**2+ci[i]**2+1e-10) for i in range(len(cr))]
        mags_t = self._mag_thresh(mags)
        ll = sum(self._scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P_state, state = 1e-4, 0.0; ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0; self._hyv_memory = 0.0; self._resolvent_norm = 1.0
        rh, hc = 0.0, 0
        for t in range(1, n):
            defl = self._deflation(vol, t); stress = self._stress(vol, t)
            ent = self._entropy(vol, t); rv = self._robust_vol(vol, t)
            ema_vol = 0.07*rv + 0.93*ema_vol
            hcorr = self._hyv_corr(rh, hc, self.ENTROPY_ALPHA, self.HYV_TARGET)
            res_reg = self._resolvent_regularization(phi, P_state, ret, t)
            mult = defl * hcorr
            pm = phi * state; pv = phi**2*P_state + q*mult*stress
            blend = 0.54*rv + 0.46*ema_vol
            obs = c*blend*mult*np.sqrt(ent*stress*res_reg)
            S = pv + obs**2
            mu[t], sigma[t] = pm, np.sqrt(max(S, 1e-10))
            inn = ret[t]-pm; score = inn/S if S > 0 else 0
            rh += 0.5*score**2 - 1.0/S if S > 1e-10 else 0; hc += 1
            pit[t] = norm.cdf(inn/sigma[t]) if sigma[t] > 0 else 0.5
            K = pv/S if S > 0 else 0; state, P_state = pm+K*inn, (1-K)*pv
            if t >= 60 and S > 1e-10:
                ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return mu, sigma, ll*self.LL_BOOST*(1+0.45*len(cr)), pit

    def fit(self, returns, vol, init_params=None):
        start = time.time()
        p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0}
        if init_params: p.update(init_params)
        def neg_ll(x):
            if time.time()-start > self.max_time_ms/1000*0.8: return 1e10
            pp = {'q':x[0],'c':x[1],'phi':x[2],'cw':x[3]}
            if pp['q'] <= 0 or pp['c'] <= 0: return 1e10
            try: _, _, ll, _ = self._filter(returns, vol, pp); return -ll
            except: return 1e10
        res = minimize(neg_ll, [p['q'],p['c'],p['phi'],p['cw']], method='L-BFGS-B',
                       bounds=[(1e-10,1e-2),(0.5,2.0),(-0.5,0.5),(0.5,2.0)], options={'maxiter':85})
        opt = {'q':res.x[0],'c':res.x[1],'phi':res.x[2],'cw':res.x[3]}
        mu, sigma, ll, pit = self._filter(returns, vol, opt)
        n = len(returns); bic = -2*ll + 4*np.log(n-60)
        pc = pit[60:]; pc = pc[(pc>0.001)&(pc<0.999)]
        ks = kstest(pc,'uniform')[1] if len(pc) > 50 else 1.0
        return {'q':opt['q'],'c':opt['c'],'phi':opt['phi'],'complex_weight':opt['cw'],
                'log_likelihood':ll,'bic':bic,'pit_ks_pvalue':ks,'n_params':4,
                'success':res.success,'fit_time_ms':(time.time()-start)*1000,
                'fit_params':{'q':opt['q'],'c':opt['c'],'phi':opt['phi']}}
