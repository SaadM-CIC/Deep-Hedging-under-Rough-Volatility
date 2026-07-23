import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.signal import fftconvolve

class RBergomiModel:
    """Modèle rBergomi (Bayer-Friz-Gatheral 2016), simulation exacte par Cholesky.

    v_t = xi0 * exp(eta*sqrt(2H)*Z_hat_t - 0.5*eta^2*t^(2H))
    dS_t = sqrt(v_t) S_t dW2_t,  dW2 = rho*dW1 + sqrt(1-rho^2)*dW_perp
    Z_hat_t = int_0^t (t-s)^(H-0.5) dW1_s  (processus de Volterra / Riemann-Liouville fBM)
    """

    def __init__(self, H: float, eta: float, rho: float, xi0: float):
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0

    def _joint_covariance(self, t_grid: NDArray) -> NDArray:
        """Covariance jointe de (Z_hat_t1..n, W1_t1..n), matrice (2n, 2n).

        Cov(W1_t, W1_s) = min(t,s)
        Cov(Z_hat_t, W1_s) = (t^(a+1) - (t - min(t,s))^(a+1)) / (a+1)
        Cov(Z_hat_t, Z_hat_s) = min(t,s)^(2H) * int_0^1 (max(t,s)/min(t,s) - x)^a (1-x)^a dx
        """
        a = self.H - 0.5
        n = len(t_grid)

        cov_ww = np.minimum(t_grid[:, None], t_grid[None, :])

        Ti = t_grid[:, None]
        Sj = t_grid[None, :]
        m = np.minimum(Ti, Sj)
        cov_zw = (Ti ** (a + 1) - (Ti - m) ** (a + 1)) / (a + 1)

        n = len(t_grid)
        cov_zz = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = self._cov_zz_entry(t_grid[i], t_grid[j])
                cov_zz[i, j] = val
                cov_zz[j, i] = val

        cov = np.zeros((2 * n, 2 * n))
        cov[:n, :n] = cov_zz
        cov[:n, n:] = cov_zw
        cov[n:, :n] = cov_zw.T
        cov[n:, n:] = cov_ww
        return cov

    def simulate_exact(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Simulation exacte (Cholesky) de (S_t, v_t) sous rBergomi.

        Coûteux en O(n^2) mémoire / O(n^3) calcul — référence lente mais correcte
        pour valider le Hybrid scheme (Phase 5.2).
        """
        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, T, n_steps + 1)
        t_interior = t_grid[1:]
        n = len(t_interior)

        cov = self._joint_covariance(t_interior)
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(2 * n))

        Z_std = rng.standard_normal(size=(n_paths, 2 * n))
        joint = Z_std @ L.T
        Z_hat = joint[:, :n]
        W1 = joint[:, n:]

        Z_hat = np.concatenate([np.zeros((n_paths, 1)), Z_hat], axis=1)
        W1 = np.concatenate([np.zeros((n_paths, 1)), W1], axis=1)

        dt = T / n_steps
        dW1 = np.diff(W1, axis=1)
        dW_perp = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        rho = self.rho
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp

        v = self.xi0 * np.exp(
            self.eta * np.sqrt(2 * self.H) * Z_hat - 0.5 * self.eta**2 * t_grid ** (2 * self.H)
        )

        log_S = np.empty((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(S0)
        for i in range(n_steps):
            log_S[:, i + 1] = log_S[:, i] + np.sqrt(v[:, i]) * dW2[:, i] - 0.5 * v[:, i] * dt

        return t_grid, np.exp(log_S), v
    def _cov_zz_entry(self, t: float, s: float) -> float:
        """Cov(Z_hat_t, Z_hat_s) exacte : forme fermée sur la diagonale (t=s),
        quadrature avec poids algébrique (scipy) pour gérer la singularité
        intégrable en x=1 hors diagonale."""
        a = self.H - 0.5
        S, T = min(t, s), max(t, s)
        if S < 1e-14:
            return 0.0
        if (T - S) < 1e-14:
            return S ** (2 * self.H) / (2 * self.H)
        ratio = T / S
        integral, _ = quad(lambda x: (ratio - x) ** a, 0, 1, weight="alg", wvar=(0, a))
        return S ** (2 * self.H) * integral



    @staticmethod
    def _b_star(k: NDArray, a: float) -> NDArray:
        """Points de discretisation optimaux du noyau (Bennedsen-Lunde-Pakkanen 2017)."""
        return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)

    def simulate_hybrid(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Hybrid scheme (kappa=1) : O(n log n) via FFT, contre O(n^3) pour
        simulate_exact (Phase 5.1). kappa=1 traite le terme le plus recent
        exactement (paire correlee via Cholesky 2x2 avec dW1), le reste par
        convolution du noyau discretise aux points optimaux b_star.
        """
        rng = np.random.default_rng(seed)
        a = self.H - 0.5
        dt = T / n_steps
        n = n_steps

        # --- paire exacte (dW1_i, Y_near_i), kappa=1 ---
        cov = np.array([
            [dt, dt ** (a + 1) / (a + 1)],
            [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)],
        ])
        L = np.linalg.cholesky(cov + 1e-14 * np.eye(2))
        Z = rng.standard_normal(size=(n_paths, n_steps, 2))
        joint = Z @ L.T
        dW1 = joint[:, :, 0]
        Y_near = joint[:, :, 1]

        # --- terme lointain (k>=2) : convolution FFT ---
        Y_far = np.zeros((n_paths, n_steps + 1))
        if n_steps >= 2:
            k_far = np.arange(2, n_steps + 1)
            weights = (self._b_star(k_far, a) * dt) ** a
            full_conv = fftconvolve(dW1, weights[None, :], mode="full", axes=1)
            Y_far[:, 2:] = full_conv[:, : n_steps - 1]

        Z_hat = np.zeros((n_paths, n_steps + 1))
        Z_hat[:, 1:] = np.sqrt(2 * self.H) * (Y_near + Y_far[:, 1:])

        t_grid = np.linspace(0.0, T, n_steps + 1)
        dW_perp = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        rho = self.rho
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp

        v = self.xi0 * np.exp(
            self.eta * Z_hat - 0.5 * self.eta**2 * t_grid ** (2 * self.H)
        )

        log_S = np.empty((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(S0)
        for i in range(n_steps):
            log_S[:, i + 1] = log_S[:, i] + np.sqrt(v[:, i]) * dW2[:, i] - 0.5 * v[:, i] * dt

        return t_grid, np.exp(log_S), v