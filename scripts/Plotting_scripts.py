import json
import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from tqdm import tqdm
from icarogw import wrappers as wps
import pickle


rw = wps.rateevolution_Madau()

colbBlue = "#4477AA"
colbRed = "#EE6677"
colbGreen = "#228833"
colbYellow = "#CCBB44"
colbCyan = "#66CCEE"
colbPurple = "#AA3377"
colbGray = "#BBBBBB"

color_palette = [
    colbBlue,
    colbRed,
    colbGreen,
    colbYellow,
    colbCyan,
    colbPurple,
    colbGray,
]

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

quantile_ranges = 0.997
levels_corner = [0.5, 0.9]  # [0.68, 0.95, 0.997]

list_params_HBH = [
    "chi_max_1g_HBH",
    # "mean_chi_2g_HBH",
    "sigma_chi_2g_HBH",
    # "f_1g_m_0_HBH",
    # "f_1g_m_inf_HBH",
    "f_1g_mt_HBH",
    "f_1g_delta_mt_HBH",
]

list_params_AGN = [
    "chi_max_1g_AGN",
    # "mean_chi_2g_AGN",
    "sigma_chi_2g_AGN",
    # "f_1g_m_0_AGN",
    # "f_1g_m_inf_AGN",
    "f_1g_mt_AGN",
    "f_1g_delta_mt_AGN",
    "delta_0_AGN",
    "delta_dot_0_AGN",
]

list_params_IBH = [
    "delta_0_IBH",
    "delta_dot_0_IBH",
    "chi_ibh_0_IBH",
    "chi_dot_ibh_0_IBH",
]

list_params_PBH = ["z_co_PBH", "sigma_chi_pbh_PBH"]

list_of_params_masses_z = [
    "alpha",
    "beta",
    "mmin",
    "mmax",
    "delta_m",
    "mu_g_low",
    "sigma_g_low",
    "mu_g_high",
    "sigma_g_high",
    "lambda_g",
    "lambda_g_low",
    "gamma",
    "kappa",
    "zp",
]

latex_list_of_params_masses_z = [
    r"$\alpha$",
    r"$\beta$",
    r"$m_{\rm min}$",
    r"$m_{\rm max}$",
    r"$\delta m$",
    r"$\mu_{g,{\rm low}}$",
    r"$\sigma_{g,{\rm low}}$",
    r"$\mu_{g,{\rm high}}$",
    r"$\sigma_{g,{\rm high}}$",
    r"$\lambda_g$",
    r"$\lambda_{g,{\rm low}}$",
    r"$\gamma$",
    r"$\kappa$",
    r"$z_p$",
]


list_of_params_spins = list_params_IBH + list_params_HBH + list_params_PBH


def read_pos_list_parameters(filename, list_of_params):

    with open(filename, "r") as f:
        posterior_data = json.load(f)

    pos = posterior_data["posterior"]

    pos = pd.DataFrame(pos)

    pos_corner_ready = np.vstack(
        [np.array(pos.loc[p, "content"]) for p in list_of_params]
    ).T

    return pos_corner_ready


def read_pos_rates(filename, keys, logs=False, total=False):

    with open(filename, "r") as f:
        posterior_data = json.load(f)

    pos = pd.DataFrame(posterior_data["posterior"])

    # Extract rates according to the keys
    rates = [np.array(pos.loc[key, "content"]) for key in keys]

    # Compute total if requested
    if total:
        Rtotal = np.sum(rates, axis=0)
        if logs:
            Rtotal = np.log10(Rtotal)
        pos_rates = np.vstack([Rtotal] + rates).T
    else:
        if logs:
            rates = [np.log10(r) for r in rates]
        pos_rates = np.vstack(rates).T

    return pos_rates


def normalized_pdf(log_pdf_func, integrate_over: str, **kwargs):

    x = np.array(kwargs[integrate_over])
    pdf_unnorm = np.exp(log_pdf_func(**kwargs))

    norm = simpson(pdf_unnorm, x=x, axis=0)
    return pdf_unnorm / norm


###################### Posterior predictive scripts ######################

IBH_sw = wps.spinprior_IBH()
HBH_sw = wps.spinprior_HBH()
PBH_sw = wps.spinprior_PBH_smeared()

xplot = np.linspace(0, 1, 200)


def plot_chi_mass_distribution(pos_plot, channels, name_plot, masses):

    # parameter sizes per channel
    param_sizes = {"IBH": 4, "HBH": 4, "PBH": 2}

    fig, axes = plt.subplots(len(masses), 1, figsize=(3, 15), sharex=True)

    for ax, m in zip(axes, masses):
        pdfs = {ch: [] for ch in channels}

        pdfs_mix = []

        for theta in pos_plot[:-1]:
            # --- split parameters and rates ---
            n_params = sum(param_sizes[ch] for ch in channels)
            n_rates = len(channels)

            params = theta[:n_params]
            rates_vec = theta[n_params : n_params + n_rates]

            # map channel -> params
            params_dict = {}
            idx = 0
            for ch in channels:
                params_dict[ch] = params[idx : idx + param_sizes[ch]]
                idx += param_sizes[ch]

            # map channel -> rate fraction
            rates = dict(zip(channels, rates_vec))
            R_tot = sum(rates.values())
            fractions = {ch: rates[ch] / R_tot for ch in channels}

            # --- evaluate PDFs ---
            if "IBH" in channels:
                p = params_dict["IBH"]
                IBH_sw.update(
                    delta_0=p[0],
                    delta_dot_0=p[1],
                    chi_ibh_0=p[2],
                    chi_dot_ibh_0=p[3],
                )
                pdfs["IBH"].append(
                    normalized_pdf(
                        IBH_sw.log_pdf,
                        "chi_1",
                        chi_1=xplot,
                        chi_2=1,
                        cos_t_1=1,
                        cos_t_2=1,
                        mass_1_source=m,
                        mass_2_source=m,
                    )
                )

            if "HBH" in channels:
                p = params_dict["HBH"]
                HBH_sw.update(
                    chi_max_1g=p[0],
                    sigma_chi_2g=p[1],
                    f_1g_mt=p[2],
                    f_1g_delta_mt=p[3],
                    mean_chi_2g=0.69,
                    f_1g_m_0=1,
                    f_1g_m_inf=0,
                )
                pdfs["HBH"].append(
                    normalized_pdf(
                        HBH_sw.log_pdf,
                        "chi_1",
                        chi_1=xplot,
                        chi_2=1,
                        cos_t_1=1,
                        cos_t_2=1,
                        mass_1_source=m,
                        mass_2_source=m,
                    )
                )

            if "PBH" in channels:
                p = params_dict["PBH"]
                PBH_sw.update(
                    z_co=p[0],
                    sigma_chi_pbh=p[1],
                )
                pdfs["PBH"].append(
                    normalized_pdf(
                        PBH_sw.log_pdf,
                        "chi_1",
                        chi_1=xplot,
                        chi_2=1,
                        cos_t_1=1,
                        cos_t_2=1,
                        mass_1_source=m,
                        mass_2_source=m,
                    )
                )

            # --- mixture ---
            pdf_mix = sum(fractions[ch] * pdfs[ch][-1] for ch in channels)
            pdfs_mix.append(pdf_mix)

        # --- convert arrays and compute averages ---
        mean_curves = {}
        for ch in channels:
            pdfs[ch] = np.array(pdfs[ch])
            mean_curves[ch] = np.mean(pdfs[ch], axis=0)

        pdfs_mix = np.array(pdfs_mix)
        mean_mix = np.mean(pdfs_mix, axis=0)
        lower_mix = np.percentile(pdfs_mix, 5, axis=0)
        upper_mix = np.percentile(pdfs_mix, 95, axis=0)

        # normalize y scale
        ymax = upper_mix.max()
        ax.set_ylim(0, 1.05 * ymax)

        if int(m / 10 - 1) == 0:
            for ch in channels:
                ax.plot(
                    xplot,
                    mean_curves[ch],
                    lw=1.5,
                    color=color_palette[{"IBH": 1, "HBH": 2, "PBH": 3}[ch]],
                    label=ch,
                )
            ax.plot(xplot, mean_mix, color="black", lw=2, label="Mixture")
            ax.fill_between(xplot, lower_mix, upper_mix, color="black", alpha=0.3)
            ax.legend(loc="upper right")
        else:
            for ch in channels:
                ax.plot(
                    xplot,
                    mean_curves[ch],
                    lw=1.5,
                    color=color_palette[{"IBH": 1, "HBH": 2, "PBH": 3}[ch]],
                )
            ax.plot(xplot, mean_mix, color="black", lw=2)
            ax.fill_between(xplot, lower_mix, upper_mix, color="black", alpha=0.3)

        ax.set_ylabel(f"{m} $M_\\odot$")
        ax.set_xlim(0, 1)

    axes[-1].set_xlabel(r"$\chi$")
    fig.suptitle(r"" + " + ".join(channels), fontsize=16)
    plt.tight_layout()
    plt.savefig("Plots/GWTC-4_chi_vs_mass_" + name_plot + ".pdf", dpi=300)
    plt.close(fig)


xplot_ct = np.linspace(-1, 1, 200)


def plot_costheta_mass_distribution(pos_plot, channels, name_plot, masses):

    # parameter sizes per channel
    param_sizes = {"IBH": 4, "HBH": 4, "PBH": 2}

    fig, axes = plt.subplots(len(masses), 1, figsize=(3, 15), sharex=True)

    for ax, m in zip(axes, masses):
        pdfs = {ch: [] for ch in channels}
        pdfs_mix = []

        for theta in pos_plot[:-1]:
            # --- split parameters and rates ---
            n_params = sum(param_sizes[ch] for ch in channels)
            n_rates = len(channels)

            params = theta[:n_params]
            rates_vec = theta[n_params : n_params + n_rates]

            # map channel -> params
            params_dict = {}
            idx = 0
            for ch in channels:
                params_dict[ch] = params[idx : idx + param_sizes[ch]]
                idx += param_sizes[ch]

            # map channel -> rate fraction
            rates = dict(zip(channels, rates_vec))
            R_tot = sum(rates.values())
            fractions = {ch: rates[ch] / R_tot for ch in channels}

            # --- evaluate PDFs ---
            if "IBH" in channels:
                p = params_dict["IBH"]
                IBH_sw.update(
                    delta_0=p[0],
                    delta_dot_0=p[1],
                    chi_ibh_0=p[2],
                    chi_dot_ibh_0=p[3],
                )
                pdfs["IBH"].append(
                    normalized_pdf(
                        IBH_sw.log_pdf,
                        "cos_t_1",
                        chi_1=1,
                        chi_2=1,
                        cos_t_1=xplot_ct,
                        cos_t_2=1,
                        mass_1_source=m,
                        mass_2_source=m,
                    )
                )

            if "HBH" in channels:
                # flat in cos θ
                pdfs["HBH"].append(np.full_like(xplot_ct, 0.5))

            if "PBH" in channels:
                # flat in cos θ
                pdfs["PBH"].append(np.full_like(xplot_ct, 0.5))

            # --- mixture ---
            pdf_mix = sum(fractions[ch] * pdfs[ch][-1] for ch in channels)
            pdfs_mix.append(pdf_mix)

        # --- convert arrays and compute averages ---
        mean_curves = {}
        for ch in channels:
            pdfs[ch] = np.array(pdfs[ch])
            mean_curves[ch] = np.mean(pdfs[ch], axis=0)

        pdfs_mix = np.array(pdfs_mix)
        mean_mix = np.mean(pdfs_mix, axis=0)
        lower_mix = np.percentile(pdfs_mix, 5, axis=0)
        upper_mix = np.percentile(pdfs_mix, 95, axis=0)

        # normalize y scale
        ymax = upper_mix.max()
        ax.set_ylim(0, np.max([1.05 * ymax, 1]))

        if int(m / 10 - 1) == 0:
            for ch in channels:
                ax.plot(
                    xplot_ct,
                    mean_curves[ch],
                    lw=1.5,
                    color=color_palette[{"IBH": 1, "HBH": 2, "PBH": 3}[ch]],
                    label=ch,
                )
            ax.plot(xplot_ct, mean_mix, color="black", lw=2, label="Mixture")
            ax.fill_between(xplot_ct, lower_mix, upper_mix, color="black", alpha=0.3)
            ax.legend(loc="upper right")
        else:
            for ch in channels:
                ax.plot(
                    xplot_ct,
                    mean_curves[ch],
                    lw=1.5,
                    color=color_palette[{"IBH": 1, "HBH": 2, "PBH": 3}[ch]],
                )
            ax.plot(xplot_ct, mean_mix, color="black", lw=2)
            ax.fill_between(xplot_ct, lower_mix, upper_mix, color="black", alpha=0.3)

        ax.set_ylabel(f"{m} $M_\\odot$")
        ax.set_xlim(-1, 1)

    axes[-1].set_xlabel(r"$\cos \theta$")
    fig.suptitle(r"" + " + ".join(channels), fontsize=16)
    plt.tight_layout()
    plt.savefig("Plots/GWTC-4_costheta_vs_mass_" + name_plot + ".pdf", dpi=300)
    plt.close(fig)


# def read_pos_rates(filename, logs):

#     with open(filename, "r") as f:
#         posterior_data = json.load(f)

#     pos = posterior_data["posterior"]

#     pos = pd.DataFrame(pos)

#     R0_IBH = np.array(pos.loc["R0_IBH", "content"])
#     R0_HBH = np.array(pos.loc["R0_HBH", "content"])
#     R0_PBH = np.array(pos.loc["R0_PBH", "content"])

#     Rtotal = R0_IBH + R0_HBH + R0_PBH

#     if not logs:
#         R0_IBH_rescaled = R0_IBH
#         R0_HBH_rescaled = R0_HBH
#         R0_PBH_rescaled = R0_PBH
#         R_total_rescaled = Rtotal

#     else:
#         R0_IBH_rescaled = np.log10(R0_IBH)
#         R0_HBH_rescaled = np.log10(R0_HBH)
#         R0_PBH_rescaled = np.log10(R0_PBH)
#         R_total_rescaled = np.log10(Rtotal)

#     pos_rates = np.vstack(
#         [R_total_rescaled, R0_IBH_rescaled, R0_HBH_rescaled, R0_PBH_rescaled]
#     ).T

#     return pos_rates


#### unchecked functions below ####

# def read_pos_rates_fraction(filename, logs):

#     with open(filename, "r") as f:
#         posterior_data = json.load(f)

#     pos = posterior_data["posterior"]

#     pos = pd.DataFrame(pos)

#     R0_IBH = np.array(pos.loc["R0_IBH", "content"])
#     R0_HBH = np.array(pos.loc["R0_HBH", "content"])
#     R0_PBH = np.array(pos.loc["R0_PBH", "content"])

#     Rtotal = R0_IBH + R0_HBH + R0_PBH

#     if not logs:
#         R0_IBH_rescaled = R0_IBH / Rtotal
#         R0_HBH_rescaled = R0_HBH / Rtotal
#         R0_PBH_rescaled = R0_PBH / Rtotal
#         R_total_rescaled = Rtotal

#     else:
#         R0_IBH_rescaled = np.log10(R0_IBH) - np.log10(Rtotal)
#         R0_HBH_rescaled = np.log10(R0_HBH) - np.log10(Rtotal)
#         R0_PBH_rescaled = np.log10(R0_PBH) - np.log10(Rtotal)
#         R_total_rescaled = np.log10(Rtotal)

#     pos_rates = np.vstack(
#         [R_total_rescaled, R0_IBH_rescaled, R0_HBH_rescaled, R0_PBH_rescaled]
#     ).T

#     return pos_rates


# def read_pos_params_rate(filename, list_of_params, logs):

#     with open(filename, "r") as f:
#         posterior_data = json.load(f)

#     pos = posterior_data["posterior"]

#     pos = pd.DataFrame(pos)

#     R0_IBH = np.array(pos.loc["R0_IBH", "content"])
#     R0_HBH = np.array(pos.loc["R0_HBH", "content"])
#     R0_PBH = np.array(pos.loc["R0_PBH", "content"])

#     Rtotal = R0_IBH + R0_HBH + R0_PBH

#     if not logs:
#         R0_IBH_rescaled = R0_IBH / Rtotal
#         R0_HBH_rescaled = R0_HBH / Rtotal
#         R0_PBH_rescaled = R0_PBH / Rtotal
#         R_total_rescaled = Rtotal

#     else:
#         R0_IBH_rescaled = np.log10(R0_IBH) - np.log10(Rtotal)
#         R0_HBH_rescaled = np.log10(R0_HBH) - np.log10(Rtotal)
#         R0_PBH_rescaled = np.log10(R0_PBH) - np.log10(Rtotal)
#         R_total_rescaled = np.log10(Rtotal)

#     pos_rates = np.vstack(
#         [np.array(pos.loc[p, "content"]) for p in list_of_params]
#         + [R_total_rescaled, R0_IBH_rescaled, R0_HBH_rescaled, R0_PBH_rescaled]
#     ).T

#     return pos_rates


# def credible_interval(samples, cl=0.9):
#     lower = (1 - cl) / 2 * 100
#     upper = (1 + cl) / 2 * 100
#     return (
#         np.percentile(samples, lower),
#         np.median(samples),
#         np.percentile(samples, upper),
#     )
