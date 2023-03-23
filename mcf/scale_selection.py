import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from skimage.feature import peak_local_max


def select_scales_gaps(
    deaths, log_times, threshold_abs=0.2, min_gap_width=0.2, with_plot=False
):

    # convert to array
    deaths = np.asarray(list(deaths))

    # drop duplicates
    deaths = np.unique(deaths)
    # replace inf with max time
    deaths[deaths == np.inf] = log_times[-1]
    # sort
    deaths.sort()

    # Compute differences to next death time
    diff_deaths = deaths[1:] - deaths[:-1]  ## Is this correct?
    diff_deaths = np.append(diff_deaths, 0)

    # Find local maxima
    local_max_ind = peak_local_max(diff_deaths, threshold_abs=threshold_abs).flatten()
    local_max_ind.sort()

    # compute gap width between max and succesor
    gap_width = np.abs(deaths[local_max_ind] - deaths[local_max_ind + 1])

    # apply minimal gap width
    local_max_ind = local_max_ind[gap_width > min_gap_width]
    gap_width = gap_width[gap_width > min_gap_width]

    # find indices of local max in log_times and of their succesors
    left_gap = np.asarray(
        [
            np.argwhere(log_times == deaths[local_max_ind[i]]).flatten()[0]
            for i in range(len(local_max_ind))
        ]
    )
    right_gap = np.asarray(
        [
            np.argwhere(log_times == deaths[local_max_ind[i] + 1]).flatten()[0]
            for i in range(len(local_max_ind))
        ]
    )

    # the optimal scales lie in the middle of the gaps
    optimal_scales = (left_gap + right_gap) // 2

    if with_plot:

        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.plot(deaths, diff_deaths, label="Difference to successor")
        ax.scatter(
            deaths[local_max_ind],
            diff_deaths[local_max_ind],
            color="green",
            label="Left gap",
        )
        ax.scatter(
            deaths[local_max_ind + 1],
            diff_deaths[local_max_ind + 1],
            color="lightgreen",
            label="Right gap",
        )

        if len(optimal_scales) > 0:
            ax.vlines(
                log_times[optimal_scales],
                0,
                diff_deaths.max(),
                color="gold",
                label="Optimal scales",
            )

        ax.set(xlabel="Deaths", ylabel="Difference")
        ax.legend()
        plt.show()

        return optimal_scales, gap_width, ax
    else:
        return optimal_scales, gap_width


def select_scales_density(
    death_count,
    birth_count,
    min_death_count=1,
    max_birth_count=1,
    log_times=[],
    with_plot=False,
):

    # compute death density
    death_total = np.sum(death_count)
    death_density = death_count / death_total

    # compute birth density
    birth_total = np.sum(birth_count)
    birth_density = birth_count / birth_total

    # PARAM1: define min height of peaks
    min_death_density = min_death_count / death_total

    # find peaks with height above threshold
    death_peaks, _ = find_peaks(death_density, height=min_death_density)

    # the last peak is selected
    selected_scales = [death_peaks[-1]]

    for i in range(len(death_peaks) - 1, -1, -1):
        # only add the previous peak if it is not a direct neighbour
        if selected_scales[-1] - death_peaks[i] > 1:
            selected_scales.append(death_peaks[i])

    selected_scales = np.sort(np.asarray(selected_scales))

    # PARAM 2: define tolerance for birth density
    max_birth_density = max_birth_count / birth_total

    # remove scales that have birth density above threshold
    selected_scales = selected_scales[
        birth_density[selected_scales] <= max_birth_density
    ]

    # find all death peaks
    death_peaks_all, _ = find_peaks(death_density, height=0)

    # as long as birth density is not above threshold, move peaks to right
    for i, s in enumerate(selected_scales):

        # compute first index to the right of s such that birth density bigger than threshold
        ind_no_birth = s + np.argmax(birth_density[s:] > max_birth_density)

        # obtain largest peak such that birth density is smaller than threshold
        if ind_no_birth > s:
            s_new = np.max(death_peaks_all[death_peaks_all < ind_no_birth])
            # replace peaks
            selected_scales[i] = s_new

    selected_scales = np.unique(selected_scales)

    if with_plot:
        if len(log_times) == 0:
            log_times = np.arange(len(death_count))
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.plot(
            log_times[1:],
            birth_density[1:-1],
            ls="-",
            alpha=0.7,
            label="Birth density",
            color="C1",
        )
        ax.plot(log_times[1:], death_density[1:-1], label="Death density")
        ax.plot(
            log_times[selected_scales],
            death_density[selected_scales],
            "x",
            label="Selected scale",
            color="gold",
        )
        ax.set(xlabel=r"$\log(t)$", ylabel="density")
        ax.legend()
        plt.show()

        return selected_scales, ax

    else:
        return selected_scales

