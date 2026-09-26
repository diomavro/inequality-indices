"""
Inequality Indices
==================

A module implementing standard inequality indices used in the economics
literature on income and wealth distribution. Each function takes a 1-D
numpy array of positive incomes and returns a scalar measure of inequality.

References
----------
Cowell, F. A. (2011). *Measuring Inequality* (3rd ed.). Oxford University Press.
Atkinson, A. B. (1970). "On the Measurement of Inequality." *Journal of
    Economic Theory*, 2(3), 244–263.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def variance(y: npt.ArrayLike) -> float:
    r"""Population variance of incomes.

    .. math::
        V(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mu)^2

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Population variance.
    """
    y = np.asarray(y, dtype=np.float64)
    return float(np.var(y, ddof=0))


def cv(y: npt.ArrayLike) -> float:
    r"""Coefficient of variation.

    .. math::
        CV(y) = \frac{\sigma}{\mu}

    where :math:`\sigma` is the population standard deviation.

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Coefficient of variation.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.mean(y)
    return float(np.std(y, ddof=0) / mu)


def variance_of_logs(y: npt.ArrayLike) -> float:
    r"""Variance of logarithms.

    .. math::
        VL(y) = \frac{1}{n} \sum_{i=1}^{n} (\ln y_i - \overline{\ln y})^2

    Parameters
    ----------
    y : array_like
        Income distribution (strictly positive values).

    Returns
    -------
    float
        Variance of log-incomes.
    """
    y = np.asarray(y, dtype=np.float64)
    log_y = np.log(y)
    return float(np.var(log_y, ddof=0))


def gini(y: npt.ArrayLike) -> float:
    r"""Gini coefficient (double-sum formula).

    .. math::
        G = \frac{1}{2 n^2 \mu} \sum_{i=1}^{n} \sum_{j=1}^{n} |y_i - y_j|

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Gini coefficient in [0, 1).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    mu = np.mean(y)
    diff_sum = np.sum(np.abs(y[:, None] - y[None, :]))
    return float(diff_sum / (2.0 * n * n * mu))


def ge_alpha(y: npt.ArrayLike, alpha: float) -> float:
    r"""Generalized Entropy index with parameter :math:`\alpha`.

    .. math::
        GE(\alpha) = \frac{1}{\alpha(\alpha - 1)}
        \left[
            \frac{1}{n} \sum_{i=1}^{n}
            \left(\frac{y_i}{\mu}\right)^{\alpha} - 1
        \right]

    Special cases:

    - :math:`\alpha = 0` (Mean Log Deviation):
      :math:`GE(0) = \frac{1}{n}\sum \ln(\mu / y_i)`
    - :math:`\alpha = 1` (Theil index):
      :math:`GE(1) = \frac{1}{n}\sum \frac{y_i}{\mu} \ln\!\left(\frac{y_i}{\mu}\right)`

    Parameters
    ----------
    y : array_like
        Income distribution (strictly positive values).
    alpha : float
        Sensitivity parameter.

    Returns
    -------
    float
        Generalized Entropy index.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    mu = np.mean(y)
    ratios = y / mu

    if np.isclose(alpha, 0.0):
        # Mean Log Deviation
        return float(np.mean(np.log(mu / y)))
    elif np.isclose(alpha, 1.0):
        # Theil index
        return float(np.mean(ratios * np.log(ratios)))
    else:
        return float(
            (np.mean(ratios**alpha) - 1.0) / (alpha * (alpha - 1.0))
        )


def theil(y: npt.ArrayLike) -> float:
    r"""Theil index (Generalized Entropy with :math:`\alpha = 1`).

    .. math::
        T = \frac{1}{n} \sum_{i=1}^{n}
        \frac{y_i}{\mu} \ln\!\left(\frac{y_i}{\mu}\right)

    Parameters
    ----------
    y : array_like
        Income distribution (strictly positive values).

    Returns
    -------
    float
        Theil index.
    """
    return ge_alpha(y, alpha=1.0)


def mld(y: npt.ArrayLike) -> float:
    r"""Mean Log Deviation (Generalized Entropy with :math:`\alpha = 0`).

    .. math::
        MLD = \frac{1}{n} \sum_{i=1}^{n}
        \ln\!\left(\frac{\mu}{y_i}\right)

    Parameters
    ----------
    y : array_like
        Income distribution (strictly positive values).

    Returns
    -------
    float
        Mean Log Deviation.
    """
    return ge_alpha(y, alpha=0.0)


def atkinson(y: npt.ArrayLike, epsilon: float) -> float:
    r"""Atkinson index with inequality aversion parameter :math:`\varepsilon`.

    .. math::
        A_\varepsilon =
        \begin{cases}
        1 - \dfrac{1}{\mu}\left[\dfrac{1}{n}\sum_{i=1}^{n}
            y_i^{1-\varepsilon}\right]^{1/(1-\varepsilon)}
            & \varepsilon \neq 1 \\[6pt]
        1 - \dfrac{1}{\mu}\left[\prod_{i=1}^{n} y_i\right]^{1/n}
            & \varepsilon = 1
        \end{cases}

    Parameters
    ----------
    y : array_like
        Income distribution (strictly positive values).
    epsilon : float
        Inequality aversion parameter (>= 0).

    Returns
    -------
    float
        Atkinson index in [0, 1).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    mu = np.mean(y)

    if np.isclose(epsilon, 1.0):
        # Geometric mean / arithmetic mean
        geo_mean = np.exp(np.mean(np.log(y)))
        return float(1.0 - geo_mean / mu)
    else:
        y_ede = np.mean(y ** (1.0 - epsilon)) ** (1.0 / (1.0 - epsilon))
        return float(1.0 - y_ede / mu)


def zenga(y: npt.ArrayLike) -> float:
    r"""Zenga (2007) inequality index.

    .. math::
        Z(\mathbf{y}) = \frac{1}{n-1}\sum_{i=1}^{n-1}
        \left(1 - \frac{\bar{y}^{-}_i}{\bar{y}^{+}_i}\right)

    where :math:`\bar{y}^{-}_i` is the mean of the bottom *i* incomes
    and :math:`\bar{y}^{+}_i` is the mean of the top *n − i* incomes,
    both computed on the sorted distribution.

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Zenga index in [0, 1).

    References
    ----------
    Zenga, M. (2007). "Inequality Curve and Inequality Index Based on the
        Ratios Between Lower and Upper Arithmetic Means."
        *Statistica & Applicazioni*, 5(1), 3–27.
    """
    y = np.asarray(y, dtype=np.float64)
    ys = np.sort(y)
    n = len(ys)
    total = 0.0
    for i in range(1, n):
        lower_mean = np.mean(ys[:i])
        upper_mean = np.mean(ys[i:])
        total += 1.0 - lower_mean / upper_mean
    return float(total / (n - 1))


def integral_zenga(y: npt.ArrayLike) -> float:
    r"""Integral Zenga index.

    .. math::
        Z^*(\mathbf{y}) = \int_0^1
        \left(1 - \frac{M^{-}(p)}{M^{+}(p)}\right) dp

    where :math:`M^{-}(p)` and :math:`M^{+}(p)` are the conditional means
    below and above quantile :math:`p`.  Computed exactly for a discrete
    distribution via piecewise-constant quantile function.

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Integral Zenga index in [0, 1).
    """
    y = np.asarray(y, dtype=np.float64)
    ys = np.sort(y)
    n = len(ys)
    cumsum = np.cumsum(ys)
    total_sum = cumsum[-1]

    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(5)

    result = 0.0
    for k in range(1, n + 1):
        # Interval ((k-1)/n, k/n): quantile = ys[k-1] (0-indexed)
        p_lo = (k - 1) / n
        p_hi = k / n
        dp = 1.0 / n

        for node, w in zip(nodes, weights):
            p = p_lo + (node + 1) / 2 * dp
            # Integral of F^{-1} from 0 to p
            if k >= 2:
                int_below = cumsum[k - 2] / n + ys[k - 1] * (p - (k - 1) / n)
            else:
                int_below = ys[0] * p
            m_minus = int_below / p
            int_above = total_sum / n - int_below
            if int_above < 1e-15:
                # p ≈ 1: integrand approaches 1 - mu/ys[-1]
                contrib = 1.0 - np.mean(ys) / ys[-1]
            else:
                m_plus = int_above / (1 - p)
                contrib = 1.0 - m_minus / m_plus
            result += w * dp / 2 * contrib

    return float(result)


def percentile_ratio(
    y: npt.ArrayLike, p: float = 0.9, q: float = 0.1
) -> float:
    r"""Percentile ratio :math:`P_p / P_q`.

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).
    p : float, default 0.9
        Upper percentile (0 < p < 1).
    q : float, default 0.1
        Lower percentile (0 < q < 1).

    Returns
    -------
    float
        Ratio of the *p*-th to the *q*-th percentile.
    """
    y = np.asarray(y, dtype=np.float64)
    upper = float(np.percentile(y, p * 100))
    lower = float(np.percentile(y, q * 100))
    return upper / lower


def absolute_gini(y: npt.ArrayLike) -> float:
    r"""Absolute Gini: :math:`\mu \cdot G`, half the mean absolute difference.

    Translation-invariant counterpart of the Gini, in income units.
    """
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(y) * gini(y))


def kolm(y: npt.ArrayLike, kappa: float) -> float:
    r"""Kolm (1976) absolute index, :math:`\kappa > 0`.

    .. math::
        K_\kappa = \frac{1}{\kappa}\ln\Bigl[\frac{1}{n}\sum_i e^{\kappa(\mu - y_i)}\Bigr]

    Translation-invariant and bottom-sensitive; ``kappa`` is in inverse
    income units, so the index is not scale-invariant.
    """
    y = np.asarray(y, dtype=np.float64)
    z = kappa * (np.mean(y) - y)
    zmax = z.max()  # log-sum-exp for numerical stability
    return float((zmax + np.log(np.mean(np.exp(z - zmax)))) / kappa)


def zenga_finite(y: npt.ArrayLike) -> float:
    r"""Zenga's finite-population index (Zenga and Jedrzejczak, 2020, eq. 1).

    Splits only at distinct income values :math:`x_h`, weighting each split by
    the population share :math:`n_h/N` at that value:

    .. math::
        Z_N = \sum_{h} \frac{n_h}{N}\left(1 - \frac{\bar{y}(Y \le x_h)}{\bar{y}(Y > x_h)}\right)

    Replication-invariant but discontinuous where incomes tie.
    """
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = len(y)
    total = 0.0
    for v, c in zip(*np.unique(y, return_counts=True)):
        upper = y[y > v]
        if len(upper):
            total += c / n * (1.0 - y[y <= v].mean() / upper.mean())
    return float(total)
