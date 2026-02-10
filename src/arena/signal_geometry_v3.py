"""
SIGNAL GEOMETRY V3 - Pure Model Following
After analysis: Model has 52.51% accuracy, +0.28 Sharpe on AAPL naive strategy.
V3: Just follow sign(mu) with fixed size. No filtering.
"""
import numpy as np
def signal_v3(mu, sigma, returns):
    if len(mu) < 20:
        return 0.0
    direction = float(np.sign(mu[-1]))
    return direction * 0.25
class SignalEngineV3:
    def get_signal(self, mu, sigma, returns):
        return signal_v3(mu, sigma, returns)
