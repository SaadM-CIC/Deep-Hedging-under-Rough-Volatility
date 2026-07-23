import numpy as np
from numpy.typing import NDArray


class HestonModel:
    """Modèle de Heston : dS = mu*S dt + sqrt(v)*S dW1, dv = kappa(theta-v)dt + xi*sqrt(v)*dW2."""

    def __init__(self, kappa: float, theta: float, xi: float, rho: float, r: float):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r

    def simulate(
        self,
        S0: float,
        v0: float,
        mu: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        psi_crit: float = 1.5,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Simulation QE scheme (Andersen 2008), schéma central gamma1=gamma2=0.5.

        mu : drift physique du sous-jacent (distinct de self.r, utilisé pour le pricing).

        Returns
        -------
        t_grid, S, v : shape (n_steps+1,), (n_paths, n_steps+1), (n_paths, n_steps+1)
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho

        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0], v[:, 0] = S0, v0
        t_grid = np.linspace(0.0, T, n_steps + 1)
        exp_kdt = np.exp(-kappa * dt)

        for i in range(n_steps):
            v_t = v[:, i]
            m = theta + (v_t - theta) * exp_kdt
            s2 = (
                v_t * xi**2 * exp_kdt / kappa * (1 - exp_kdt)
                + theta * xi**2 / (2 * kappa) * (1 - exp_kdt) ** 2
            )
            psi = s2 / m**2
            v_next = np.empty(n_paths)

            low = psi <= psi_crit
            high = ~low

            psi_low = psi[low]
            b2 = 2 / psi_low - 1 + np.sqrt(2 / psi_low) * np.sqrt(2 / psi_low - 1)
            a = m[low] / (1 + b2)
            Z = rng.standard_normal(low.sum())
            v_next[low] = a * (np.sqrt(b2) + Z) ** 2

            psi_high = psi[high]
            p = (psi_high - 1) / (psi_high + 1)
            beta = (1 - p) / m[high]
            U = rng.uniform(size=high.sum())
            v_next[high] = np.where(U <= p, 0.0, np.log((1 - p) / (1 - U)) / beta)

            gamma1 = gamma2 = 0.5
            K0 = -rho * kappa * theta * dt / xi
            K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
            K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
            K3 = gamma1 * dt * (1 - rho**2)
            K4 = gamma2 * dt * (1 - rho**2)
            Zs = rng.standard_normal(n_paths)

            var_term = np.maximum(K3 * v_t + K4 * v_next, 0.0)
            log_S_next = (
                np.log(S[:, i]) + mu * dt + K0 + K1 * v_t + K2 * v_next + np.sqrt(var_term) * Zs
            )
            S[:, i + 1] = np.exp(log_S_next)
            v[:, i + 1] = v_next

        return t_grid, S, v

    def _phi(self, u: NDArray, j: int, S: NDArray, v: NDArray, tau: float) -> NDArray:
        """Fonction caractéristique de Heston, formulation stable ("little trap" fix,
        Albrecher et al. 2007) — évite le saut de branche du log complexe.
        """
        kappa, theta, xi, rho, r = self.kappa, self.theta, self.xi, self.rho, self.r
        b = kappa - rho * xi if j == 1 else kappa
        uj = 0.5 if j == 1 else -0.5

        d = np.sqrt((rho * xi * 1j * u - b) ** 2 - xi**2 * (2 * uj * 1j * u - u**2))
        g = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        exp_neg_dtau = np.exp(-d * tau)

        C = r * 1j * u * tau + (kappa * theta / xi**2) * (
            (b - rho * xi * 1j * u - d) * tau - 2 * np.log((1 - g * exp_neg_dtau) / (1 - g))
        )
        D = (b - rho * xi * 1j * u - d) / xi**2 * (1 - exp_neg_dtau) / (1 - g * exp_neg_dtau)
        return np.exp(C + D * v + 1j * u * np.log(S))

    def _prob_j(
        self, S: NDArray, v: NDArray, tau: float, K: float, j: int,
        n_quad: int = 64, u_max: float = 200.0,
    ) -> NDArray:
        """P1 ou P2 (Heston 1993), quadrature de Gauss-Legendre vectorisée sur tous les paths.

        Gauss-Legendre évite le point u=0 exact (singularité amovible mal gérée en
        float par une grille linéaire) et converge bien plus vite qu'un linspace naïf.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        u = 0.5 * u_max * (nodes + 1.0)          # rescale [-1,1] -> [0, u_max]
        w = 0.5 * u_max * weights

        phi = self._phi(u[None, :], j, S[:, None], v[:, None], tau)
        integrand = np.real(np.exp(-1j * u[None, :] * np.log(K)) * phi / (1j * u[None, :]))
        integral = (integrand * w[None, :]).sum(axis=1)
        return 0.5 + integral / np.pi

    def call_price(self, S: NDArray, v: NDArray, K: float, tau: float) -> NDArray:
        """Prix call vectorisé sur des arrays (S, v), tau et K scalaires (même maturité pour tous)."""
        if tau <= 1e-12:
            return np.maximum(S - K, 0.0)
        P1 = self._prob_j(S, v, tau, K, j=1)
        P2 = self._prob_j(S, v, tau, K, j=2)
        return S * P1 - K * np.exp(-self.r * tau) * P2

    def call_delta(self, S: NDArray, v: NDArray, K: float, tau: float) -> NDArray:
        """Delta Heston correct = P1 (résultat semi-analytique connu)."""
        if tau <= 1e-12:
            return (S > K).astype(float)
        return self._prob_j(S, v, tau, K, j=1)
    def call_vega(self, S: NDArray, v: NDArray, K: float, tau: float, eps: float = 1e-4) -> NDArray:
        """Vega = dC/dv, par différences finies centrées (v0 étant la variance, pas sigma)."""
        v_up = v + eps
        v_down = np.maximum(v - eps, 1e-8)
        return (self.call_price(S, v_up, K, tau) - self.call_price(S, v_down, K, tau)) / (2 * eps)

    def call_delta_min_variance(self, S: NDArray, v: NDArray, K: float, tau: float) -> NDArray:
        """Delta à variance minimale : corrige P1 de l'exposition indirecte à v via rho."""
        if tau <= 1e-12:
            return (S > K).astype(float)
        return self.call_delta(S, v, K, tau) + (self.rho * self.xi / S) * self.call_vega(S, v, K, tau)