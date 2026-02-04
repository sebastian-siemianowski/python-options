#!/usr/bin/env python
"""Quick test for Numba kernels."""
import sys
import numpy as np

print("Starting Numba test...")

try:
    from models.numba_kernels import gaussian_filter_kernel, phi_gaussian_filter_kernel, phi_student_t_filter_kernel
    print("Kernel import successful")
    
    # Test execution
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02
    vol = np.abs(np.random.randn(100)) * 0.01 + 0.01
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    
    print("Testing Gaussian kernel...")
    mu, P, ll = gaussian_filter_kernel(returns, vol, 1e-6, 1.0, 1e-4)
    print(f"  Gaussian: mu[-1]={mu[-1]:.6f}, ll={ll:.2f}")
    
    print("Testing phi-Gaussian kernel...")
    mu, P, ll = phi_gaussian_filter_kernel(returns, vol, 1e-6, 1.0, 0.7, 1e-4)
    print(f"  phi-Gaussian: mu[-1]={mu[-1]:.6f}, ll={ll:.2f}")
    
    print("Testing phi-Student-t kernel...")
    from scipy.special import gammaln
    log_g1 = float(gammaln(6.0 / 2.0))
    log_g2 = float(gammaln((6.0 + 1.0) / 2.0))
    mu, P, ll = phi_student_t_filter_kernel(returns, vol, 1e-6, 1.0, 0.5, 6.0, log_g1, log_g2, 1e-4)
    print(f"  phi-Student-t: mu[-1]={mu[-1]:.6f}, ll={ll:.2f}")
    
    print("\nSUCCESS! All Numba kernels compiled and executed correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
