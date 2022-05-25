import matplotlib.pyplot as plt
import numpy as np

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
