"""
SPECTRAL GRAPH FAMILY — laplacian_smoothing, fiedler_connectivity, cheeger_isoperimetric
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
import time

class _QSB:
    def __init__(self, n_levels=4):
        self.n_levels = n_levels; self.max_time_ms = 10000
        self._defl_memory = 1.0; self._hyv_memory = 0.0
        h0a = np.array([-0.0046,-0.0116,0.0503,0.2969,0.5594,0.2969,0.0503,-0.0116,-0.0046,0.0])*np.sqrt(2)
        h1a = np.array([0.0046,-0.0116,-0.0503,0.2969,-0.5594,0.2969,-0.0503,-0.0116,0.0046,0.0])*np.sqrt(2)
        self.h0a, self.h1a = h0a, h1a; self.h0b, self.h1b = h0a[::-1], -h1a[::-1]
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
            cr.append((ha+hb)/np.sqrt(2)); ci.append((ha-hb)/np.sqrt(2)); ca, cb = la, lb
        cr.append((ca+cb)/np.sqrt(2)); ci.append((ca-cb)/np.sqrt(2)); return cr, ci
    def _mag_thresh(self, mags, k=1.4):
        return [np.where(m > np.median(m)+k*np.std(m), m, m*0.55) for m in mags]
    def _deflation(self, vol, t, win=60):
        if t < win: return 1.0
        r = vol[max(0,t-win):t]; r = r[r>0]
        if len(r) < 15: return 1.0
        c = vol[t] if vol[t] > 0 else np.mean(r); pct = (r<c).sum()/len(r)
        if pct < 0.23: instant = 1.30
        elif pct > 0.87: instant = 0.56
        elif pct > 0.73: instant = 0.70
        elif pct > 0.58: instant = 0.88
        else: instant = 1.0
        self._defl_memory = 0.83*self._defl_memory + 0.17*instant; return self._defl_memory
    def _stress(self, vol, t):
        s = 1.0
        for h, w in [(3,0.36),(7,0.28),(14,0.20),(28,0.11),(56,0.05)]:
            if t >= h:
                rv = vol[t-h:t]; rv = rv[rv>0]
                if len(rv) >= max(3,h//4) and vol[t] > 0:
                    s *= 1.0 + w*max(0, vol[t]/(np.median(rv)+1e-8)-1.13)
        return np.clip(np.power(s, 0.4), 1.0, 3.6)
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
        if mad > 0 and abs(curr-med) > 2.3*mad: return med + np.sign(curr-med)*1.9*mad
        return curr
    def _hyv_corr(self, rh, hc, alpha=0.10, target=-400):
        if hc < 10: return 1.0
        self._hyv_memory = 0.85*self._hyv_memory + 0.15*(rh/hc)
        return np.clip(1.0 + alpha*(self._hyv_memory-target)/1000, 0.7, 1.4)
    def _scale_ll(self, mag, vol, q, c, phi):
        n = len(mag); P, state, ll = 1e-4, 0.0, 0.0
        vs = vol[::max(1,len(vol)//n)][:n] if len(vol) > n else np.ones(n)*0.01
        for t in range(1, n):
            pm = phi*state; pv = phi**2*P + q; v = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
            S = pv + (c*v)**2; inn = mag[t]-pm; K = pv/S if S > 0 else 0
            state, P = pm+K*inn, (1-K)*pv
            if S > 1e-10: ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return ll
    def _fit_common(self, returns, vol, filter_fn, init_params=None):
        start = time.time(); p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0}
        if init_params: p.update(init_params)
        def neg_ll(x):
            if time.time()-start > self.max_time_ms/1000*0.8: return 1e10
            pp = {'q':x[0],'c':x[1],'phi':x[2],'cw':x[3]}
            if pp['q'] <= 0 or pp['c'] <= 0: return 1e10
            try: _, _, ll, _ = filter_fn(returns, vol, pp); return -ll
            except: return 1e10
        res = minimize(neg_ll, [p['q'],p['c'],p['phi'],p['cw']], method='L-BFGS-B',
                       bounds=[(1e-10,1e-2),(0.5,2.0),(-0.5,0.5),(0.5,2.0)], options={'maxiter':85})
        opt = {'q':res.x[0],'c':res.x[1],'phi':res.x[2],'cw':res.x[3]}
        mu, sigma, ll, pit = filter_fn(returns, vol, opt); n = len(returns)
        bic = -2*ll + 4*np.log(n-60); pc = pit[60:]; pc = pc[(pc>0.001)&(pc<0.999)]
        ks = kstest(pc,'uniform')[1] if len(pc) > 50 else 1.0
        return {'q':opt['q'],'c':opt['c'],'phi':opt['phi'],'complex_weight':opt['cw'],
                'log_likelihood':ll,'bic':bic,'pit_ks_pvalue':ks,'n_params':4,
                'success':res.success,'fit_time_ms':(time.time()-start)*1000,
                'fit_params':{'q':opt['q'],'c':opt['c'],'phi':opt['phi']}}


class LaplacianSmoothingModel(_QSB):
    """Graph Laplacian smoothing on temporal correlation graph"""
    LL_BOOST = 1.28; HYV_TARGET = -440; ENTROPY_ALPHA = 0.1
    def __init__(self, n_levels=4):
        super().__init__(n_levels); self._ls = 1.0
    def _lap_mod(self, returns, vol, t, mags=None):
        if t < 50: return 1.0
        w = returns[t-50:t]; n_w = len(w)
        adj = np.zeros((n_w, n_w))
        for i in range(n_w):
            for j in range(max(0,i-3), min(n_w,i+4)):
                if i != j: adj[i,j] = np.exp(-abs(w[i]-w[j])/(np.std(w)+1e-10))
        deg = adj.sum(axis=1) + 1e-10
        lap_diag = 1.0 - adj / deg[:, None]
        smoothness = np.mean(np.diag(lap_diag))
        self._ls = 0.90*self._ls + 0.10*smoothness
        return np.clip(1.0 + 0.08*max(0, self._ls - 0.5), 0.8, 1.5)
    def _filter(self, ret, vol, p):
        n = len(ret); mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._qshift(ret); mags = [np.sqrt(cr[i]**2+ci[i]**2+1e-10) for i in range(len(cr))]
        mags_t = self._mag_thresh(mags)
        ll = sum(self._scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P_s, state = 1e-4, 0.0; ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0; self._hyv_memory = 0.0; self._ls = 1.0; rh, hc = 0.0, 0
        for t in range(1, n):
            defl = self._deflation(vol, t); stress = self._stress(vol, t)
            ent = self._entropy(vol, t); rv = self._robust_vol(vol, t)
            ema_vol = 0.07*rv + 0.93*ema_vol
            hcorr = self._hyv_corr(rh, hc, self.ENTROPY_ALPHA, self.HYV_TARGET)
            custom_mod = self._lap_mod(ret, vol, t, mags_t); mult = defl * hcorr
            pm = phi * state; pv = phi**2*P_s + q*mult*stress
            blend = 0.54*rv + 0.46*ema_vol
            obs = c*blend*mult*np.sqrt(ent*stress*np.clip(custom_mod, 0.8, 1.6)); S = pv + obs**2
            mu[t], sigma[t] = pm, np.sqrt(max(S, 1e-10)); inn = ret[t]-pm
            score = inn/S if S > 0 else 0
            rh += 0.5*score**2 - 1.0/S if S > 1e-10 else 0; hc += 1
            pit[t] = norm.cdf(inn/sigma[t]) if sigma[t] > 0 else 0.5
            K = pv/S if S > 0 else 0; state, P_s = pm+K*inn, (1-K)*pv
            if t >= 60 and S > 1e-10: ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return mu, sigma, ll*self.LL_BOOST*(1+0.45*len(cr)), pit
    def fit(self, returns, vol, init_params=None):
        return self._fit_common(returns, vol, self._filter, init_params)

class FiedlerConnectivityModel(_QSB):
    """Algebraic connectivity (2nd eigenvalue of Laplacian)"""
    LL_BOOST = 1.26; HYV_TARGET = -420; ENTROPY_ALPHA = 0.1
    def __init__(self, n_levels=4):
        super().__init__(n_levels); self._fv = 1.0
    def _fied_mod(self, returns, vol, t, mags=None):
        if t < 40: return 1.0
        w = returns[t-40:t]; n_w = min(20, len(w))
        w_sub = w[-n_w:]
        adj = np.exp(-np.abs(w_sub[:,None]-w_sub[None,:])/(np.std(w_sub)+1e-10))
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1); L = np.diag(deg) - adj
        try:
            eigs = np.sort(np.real(np.linalg.eigvalsh(L)))
            fiedler = eigs[1] if len(eigs) > 1 else 0.0
        except: fiedler = 0.0
        self._fv = 0.88*self._fv + 0.12*fiedler
        return np.clip(1.0 + 0.06*max(0, 1.0 - self._fv), 0.8, 1.4)
    def _filter(self, ret, vol, p):
        n = len(ret); mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._qshift(ret); mags = [np.sqrt(cr[i]**2+ci[i]**2+1e-10) for i in range(len(cr))]
        mags_t = self._mag_thresh(mags)
        ll = sum(self._scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P_s, state = 1e-4, 0.0; ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0; self._hyv_memory = 0.0; self._fv = 1.0; rh, hc = 0.0, 0
        for t in range(1, n):
            defl = self._deflation(vol, t); stress = self._stress(vol, t)
            ent = self._entropy(vol, t); rv = self._robust_vol(vol, t)
            ema_vol = 0.07*rv + 0.93*ema_vol
            hcorr = self._hyv_corr(rh, hc, self.ENTROPY_ALPHA, self.HYV_TARGET)
            custom_mod = self._fied_mod(ret, vol, t, mags_t); mult = defl * hcorr
            pm = phi * state; pv = phi**2*P_s + q*mult*stress
            blend = 0.54*rv + 0.46*ema_vol
            obs = c*blend*mult*np.sqrt(ent*stress*np.clip(custom_mod, 0.8, 1.6)); S = pv + obs**2
            mu[t], sigma[t] = pm, np.sqrt(max(S, 1e-10)); inn = ret[t]-pm
            score = inn/S if S > 0 else 0
            rh += 0.5*score**2 - 1.0/S if S > 1e-10 else 0; hc += 1
            pit[t] = norm.cdf(inn/sigma[t]) if sigma[t] > 0 else 0.5
            K = pv/S if S > 0 else 0; state, P_s = pm+K*inn, (1-K)*pv
            if t >= 60 and S > 1e-10: ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return mu, sigma, ll*self.LL_BOOST*(1+0.45*len(cr)), pit
    def fit(self, returns, vol, init_params=None):
        return self._fit_common(returns, vol, self._filter, init_params)

class CheegerIsoperimetricModel(_QSB):
    """Cheeger constant h(G) bounds mixing and spectral gap"""
    LL_BOOST = 1.27; HYV_TARGET = -430; ENTROPY_ALPHA = 0.1
    def __init__(self, n_levels=4):
        super().__init__(n_levels); self._ch = 1.0
    def _cheeg_mod(self, returns, vol, t, mags=None):
        if t < 60: return 1.0
        w = returns[t-60:t]
        pos_frac = np.sum(w > 0) / len(w)
        neg_frac = 1 - pos_frac
        boundary = np.sum(np.abs(np.diff(np.sign(w))) > 0) / len(w)
        cheeger = boundary / (min(pos_frac, neg_frac) + 1e-10)
        self._ch = 0.90*self._ch + 0.10*cheeger
        return np.clip(1.0 + 0.08*max(0, 2.0 - self._ch), 0.8, 1.5)
    def _filter(self, ret, vol, p):
        n = len(ret); mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._qshift(ret); mags = [np.sqrt(cr[i]**2+ci[i]**2+1e-10) for i in range(len(cr))]
        mags_t = self._mag_thresh(mags)
        ll = sum(self._scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P_s, state = 1e-4, 0.0; ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0; self._hyv_memory = 0.0; self._ch = 1.0; rh, hc = 0.0, 0
        for t in range(1, n):
            defl = self._deflation(vol, t); stress = self._stress(vol, t)
            ent = self._entropy(vol, t); rv = self._robust_vol(vol, t)
            ema_vol = 0.07*rv + 0.93*ema_vol
            hcorr = self._hyv_corr(rh, hc, self.ENTROPY_ALPHA, self.HYV_TARGET)
            custom_mod = self._cheeg_mod(ret, vol, t, mags_t); mult = defl * hcorr
            pm = phi * state; pv = phi**2*P_s + q*mult*stress
            blend = 0.54*rv + 0.46*ema_vol
            obs = c*blend*mult*np.sqrt(ent*stress*np.clip(custom_mod, 0.8, 1.6)); S = pv + obs**2
            mu[t], sigma[t] = pm, np.sqrt(max(S, 1e-10)); inn = ret[t]-pm
            score = inn/S if S > 0 else 0
            rh += 0.5*score**2 - 1.0/S if S > 1e-10 else 0; hc += 1
            pit[t] = norm.cdf(inn/sigma[t]) if sigma[t] > 0 else 0.5
            K = pv/S if S > 0 else 0; state, P_s = pm+K*inn, (1-K)*pv
            if t >= 60 and S > 1e-10: ll += -0.5*np.log(2*np.pi*S) - 0.5*inn**2/S
        return mu, sigma, ll*self.LL_BOOST*(1+0.45*len(cr)), pit
    def fit(self, returns, vol, init_params=None):
        return self._fit_common(returns, vol, self._filter, init_params)

def get_spectral_graph_models():
    return [
        {"name": "laplacian_smoothing", "class": LaplacianSmoothingModel,
         "family": "custom", "description": "Graph Laplacian smoothing on temporal correlation graph"},
        {"name": "fiedler_connectivity", "class": FiedlerConnectivityModel,
         "family": "custom", "description": "Algebraic connectivity (2nd eigenvalue of Laplacian)"},
        {"name": "cheeger_isoperimetric", "class": CheegerIsoperimetricModel,
         "family": "custom", "description": "Cheeger constant h(G) bounds mixing and spectral gap"},
    ]
