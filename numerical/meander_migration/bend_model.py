#!/usr/bin/env python3
"""Ikeda-Parker-Sawai (1981) linear bend theory: a RIGID (equilibrium) bed with a MIGRATING
bank.  The channel centreline y(x,t) is the only state; the bed cross-section is at its
secondary-flow equilibrium and does NOT evolve (the opposite trade-off to funwave_2d_sw, which
had a mobile bed and fixed banks).  This is the model the meander-migration question actually
needs -- the bank waveform grows and migrates down-valley.

Physics, verbatim from Ikeda, Parker & Sawai (1981) "Bend theory of river meanders. Part 1.
Linear development", J. Fluid Mech. 112, 363-377 (on disk: rossby_palooza/literature/):

  eq (11)  gamma d(y)/dt = zeta,   gamma = cos(theta)         bank migrates normal to itself
  eq (12)  zeta = E(U) u'(s,b)                                erosion rate ~ NEAR-BANK velocity
                                                              perturbation (reach mean vanishes)
  eq (16)  y_xt + 2 Cf y_t = y_xxx - Cf (A + F^2) y_xx        the linearized BEND EQUATION
  eq (17)  y = eps e^{alpha0 t} cos(kx - omega0 t)            normal mode

with A = 2.89 (alluvial secondary-flow closure, Suga 1963), F = U0/sqrt(g H0) the Froude number,
Cf the friction factor.  Lengths are scaled by the depth H0, so k = 2 pi H0 / lambda and
eps = y0/H0 (Ikeda's eq (15) discussion).  Time absorbs the bank-erodibility E0 b*.

Dispersion relation (substitute eq 17 into eq 16, y ~ e^{ikx} e^{pt}, p = alpha0 - i omega0):

  alpha0(k) = k^2 [ 2 Cf^2 (A+F^2) - k^2 ] / (k^2 + 4 Cf^2)      GROWTH rate
  omega0(k) = Cf k^3 [ (A+F^2) + 2 ]     / (k^2 + 4 Cf^2)        MIGRATION frequency

  -> bends GROW for k < Cf sqrt(2(A+F^2)); a max-growth k selects the meander wavelength.
  -> omega0 > 0 for all k -> bends ALWAYS migrate DOWNSTREAM (Ikeda's result).
  -> the driver is (A + F^2): A=2.89 (secondary-flow / vortical) vs F^2 (gravity / Froude).
     For a real river F^2 << A, so meandering is SECONDARY-FLOW driven, not a gravity wave,
     and c0 = omega0/k << sqrt(g H0).
"""
import numpy as np


class BendParams:
    def __init__(self, H0=3.0, U0=0.85, Cf=0.005, A=2.89, b=50.0, g=9.81):
        self.H0, self.U0, self.Cf, self.A, self.b, self.g = H0, U0, Cf, A, b, g
        self.F = U0 / np.sqrt(g * H0)          # Froude number
        self.F2 = self.F ** 2
        self.B = A + self.F2                    # the (A + F^2) driver in eq (16)
        self.bstar = b / H0

    # ---- dispersion relation, eq (16)/(17) --------------------------------
    def growth(self, k):
        """alpha0(k): dimensionless amplitude growth rate (even in k)."""
        return k ** 2 * (2 * self.Cf ** 2 * self.B - k ** 2) / (k ** 2 + 4 * self.Cf ** 2)

    def migr_freq(self, k):
        """omega0(k): dimensionless migration frequency (ODD in k -> coherent downstream)."""
        return self.Cf * k ** 3 * (self.B + 2.0) / (k ** 2 + 4 * self.Cf ** 2)

    def p(self, k):
        """complex growth rate p = alpha0 - i*omega0 for a mode y ~ e^{ikx} e^{pt}."""
        return self.growth(k) - 1j * self.migr_freq(k)

    def k_cut(self):
        """neutral wavenumber (alpha0 = 0); bends grow below it."""
        return self.Cf * np.sqrt(2.0 * self.B)

    def k_max(self):
        """wavenumber of MAXIMUM growth -> the selected meander wavelength."""
        kc = self.k_cut()
        kk = np.linspace(1e-4 * kc, kc, 20000)
        return kk[np.argmax(self.growth(kk))]

    def wavelength(self, k):
        """physical meander wavelength [m] from k = 2 pi H0 / lambda."""
        return 2.0 * np.pi * self.H0 / k

    def migration_speed(self, k):
        """dimensionless downstream bend migration speed c0 = omega0 / k."""
        return self.migr_freq(k) / k


def evolve(y0, dx, params, times):
    """EXACT linear evolution of eq (16) in Fourier space (constant coefficients):
        y(x,t) = IFFT[ FFT(y0)(k) * exp( p(k) t ) ]
    y0 is the initial centreline offset (dimensionless, scaled by H0) on a PERIODIC domain with
    spacing dx (also /H0).  Returns (len(times), len(y0)).  p(k) uses SIGNED k so growth (k^2,
    even) and downstream migration (k^3, odd) are both correct for the real field."""
    N = len(y0)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)      # signed dimensionless wavenumbers
    Y0 = np.fft.fft(y0)
    P = params.p(k)
    return np.array([np.real(np.fft.ifft(Y0 * np.exp(P * t))) for t in times])


def near_bank_velocity(y_series, times):
    """u'_b(x,t), the near-bank velocity perturbation that DRIVES migration.  From Ikeda eq (13)
    gamma y_t = E u'_b, so (dimensionless, gamma~1, E absorbed into t) u'_b = dy/dt.  This is the
    'u-prime' the flow contributes; outer bank erodes where u'_b > 0.  Central-differenced in t."""
    ub = np.gradient(y_series, times, axis=0)
    return ub


if __name__ == "__main__":
    # quick self-report of the selected wavelength and the gravity-vs-secondary-flow split
    for Cf in (0.003, 0.005, 0.01):
        pr = BendParams(Cf=Cf)
        km = pr.k_max()
        print(f"Cf={Cf:.3f}: F={pr.F:.3f}, F^2={pr.F2:.4f}, A={pr.A}  ->  A+F^2={pr.B:.3f} "
              f"(F^2 is {100*pr.F2/pr.B:.1f}% -> {'gravity' if pr.F2>pr.A else 'secondary-flow'}-dominated)")
        print(f"          selected k_max={km:.4f}, lambda={pr.wavelength(km):.0f} m "
              f"= {pr.wavelength(km)/(2*pr.b):.1f} W,  c0(k_max)={pr.migration_speed(km):.4f} (dimensionless)")
