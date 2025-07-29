import numpy as np

def Pchieff(chieffv, chi1, chi2, q):
    return np.where(
        abs(chieffv) <= (chi1 + chi2 * q) / (1 + q),
        (
            (
                (1 + q) ** 2
                * (
                    ((chieffv + chieffv * q - chi1 - q * chi2) ** 2 / (1 + q) ** 2)
                    ** (1 / 2)
                    - ((chieffv + chieffv * q + chi1 - q * chi2) ** 2 / (1 + q) ** 2)
                    ** (1 / 2)
                    - ((chieffv + chieffv * q - chi1 + q * chi2) ** 2 / (1 + q) ** 2)
                    ** (1 / 2)
                    + ((chieffv + chieffv * q + chi1 + q * chi2) ** 2 / (1 + q) ** 2)
                    ** (1 / 2)
                )
            )
            / (8.0 * q * chi1 * chi2)
        ),
        0.0,
    )


# PBH mass spin relation
def fa1(zco, q):
    return (
        57.8531
        - 56.4905 * q
        + 39.4605 * q**2
        - 10.5127 * q**3
        - 6.68879 * zco
        + 3.74532 * q * zco
        - 1.756 * q**2 * zco
        + 0.439529 * zco**2
        - 0.175899 * q * zco**2
        + 0.073267 * q**2 * zco**2
        - 0.00546522 * zco**3
    )


def fb1(zco, q):
    return (
        2.1468
        - 1.59262 * q
        - 1.33445 * q**2
        + 0.940219 * q**3
        - 0.365483 * zco
        + 0.248367 * q * zco
        + 0.00136971 * q**2 * zco
        + 0.0123732 * zco**2
        - 0.00313974 * q * zco**2
        - 0.00218091 * q**2 * zco**2
        - 0.000185276 * zco**3
    )


def fc1(zco, q):
    return (
        0.441418
        - 0.231674 * q
        + 2.12451 * q**2
        - 0.7873 * q**3
        - 0.0738179 * zco
        - 0.00461876 * q * zco
        - 0.120687 * q**2 * zco
        + 0.00834177 * zco**2
        - 0.00234563 * q * zco**2
        + 0.0047721 * q**2 * zco**2
        - 0.000175491 * zco**3
    )


def fa2(zco, q):
    return (
        44.322
        + 19.8378 * q
        - 33.8142 * q**2
        + 18.3605 * q**3
        - 7.27617 * zco
        - 0.680676 * q * zco
        + 0.195003 * q**2 * zco
        + 0.509837 * zco**2
        + 0.000581762 * q * zco**2
        - 0.00957243 * q**2 * zco**2
        - 0.00827027 * zco**3
    )


def fb2(zco, q):
    return (
        3.65282
        - 0.474109 * q
        - 0.199862 * q**2
        + 0.0523957 * q**3
        - 0.694442 * zco
        + 0.0737077 * q * zco
        + 0.00855668 * q**2 * zco
        + 0.035586 * zco**2
        - 0.00178022 * q * zco**2
        - 0.000212303 * q**2 * zco**2
        - 0.000630911 * zco**3
    )


def fc2(zco, q):
    return (
        -0.189439
        - 0.905386 * q
        + 1.25085 * q**2
        - 0.346207 * q**3
        + 0.128502 * zco
        + 0.0158765 * q * zco
        - 0.0447706 * q**2 * zco
        - 0.00587638 * zco**2
        - 0.000109165 * q * zco**2
        + 0.000945026 * q**2 * zco**2
        + 0.0000864602 * zco**3
    )


def chi1_analytical_fit(m1, q, zco):
    return 0.01 + np.minimum(
        0.988,
        10 ** fb1(zco, q)
        * (np.absolute(m1 - fa1(zco, q))) ** fc1(zco, q)
        * np.heaviside(m1 - fa1(zco, q), 1),
    )


def chi2_analytical_fit(m1, q, zco):
    return 0.01 + np.minimum(
        0.988,
        10 ** fb2(zco, q)
        * (np.absolute(m1 - fa2(zco, q))) ** fc2(zco, q)
        * np.heaviside(m1 - fa2(zco, q), 1),
    )


# 0: q #1: chi1 #2: chi2 #3: chieff val
def spindistPBH(zco, m1, m2, x):
    q = m2 / m1
    return Pchieff(x, chi1(m1, q, zco), chi2(m1, q, zco), q)


# distribution of chi1 and chi2. Define a Gaussian normalized to 1 for x in [0,1]
def norm_g(x, mu, sigma):
    normalization_factor = 1 / (
        0.5 * (erf((1 - mu) / (np.sqrt(2) * sigma)) + erf(mu / (np.sqrt(2) * sigma)))
    )
    gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return normalization_factor * gaussian


# define width of the Gaussian
sigma_Pchi = 0.05


def chi1_Smoothed(x, m1, q, zco):
    return norm_g(x, chi1(m1, q, zco), sigma_Pchi)


def chi2_Smoothed(x, m1, q, zco):
    return norm_g(x, chi2(m1, q, zco), sigma_Pchi)


# ABH mass spin relation
def Li2(x):
    return scipy.special.spence(1 - x)


def Pchi1G1G(x, xmax):
    chi = abs(x / xmax)
    return (
        np.where(
            chi < 1,
            np.where(
                chi < 1 / 2,
                2
                - ((4 + np.pi**2) * chi) / 2.0
                - np.log(1 - 2 * chi) * (1 - 2 * chi + chi * np.log(1 - 2 * chi))
                - 2 * chi * Li2((2 * chi) / (-1 + 2 * chi)),
                2
                - 2 * chi
                - (1 - 2 * chi + chi * np.log(2 * chi)) * np.log(-1 + 2 * chi)
                + chi * Li2(1 - 1 / (2.0 * chi))
                - chi * Li2(1 / (2.0 * chi)),
            ),
            0.0,
        )
        / xmax
    )


chif = 0.68


def Pchi1G2G(x, xmax, q):
    return Pchieff(x, chif, xmax, q)


def frac(f0, alphaM, alphaq, m1, m2):
    return max(0, min(f0 + alphaM * (m1 + m2) / 60 - alphaq * ((m2 / m1) - 0.5), 1))


def PchiDyn(m1, m2, x, xmax, f0, alphaM, alphaq):
    f = frac(f0, alphaM, alphaq, m1, m2)
    return (1 - f) * Pchi1G1G(x, xmax) + f * Pchi1G2G(x, xmax, m2 / m1)
