import argparse

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture


def expected_value_of_y_given_x(model, x_values):
    # Extract parameters from the GaussianMixture model
    weights = model.weights_  # Mixing weights
    means = model.means_  # Means of each component
    covariances = model.covariances_  # Covariances of each component

    # Prepare to store the expected values
    expected_values = np.zeros_like(x_values)

    # Iterate over each x value to calculate E[Y | X = x]
    for i, x in enumerate(x_values):
        weighted_sum = 0
        total_weight = 0

        # Compute weights and conditional means for each component
        for k in range(len(weights)):
            mean_x = means[k, 0]
            mean_y = means[k, 1]
            var_x = covariances[k][0, 0]
            cov_xy = covariances[k][0, 1]
            cov_yx = covariances[k][1, 0]
            var_y = covariances[k][1, 1]

            # Compute the density of x under the k-th Gaussian component's marginal
            density_x = (1 / np.sqrt(2 * np.pi * var_x)) * np.exp(-0.5 * ((x - mean_x) ** 2) / var_x)

            # Compute conditional mean E[Y | X = x] for the k-th component
            conditional_mean_y = mean_y + (cov_yx / var_x) * (x - mean_x)

            # Weighted sum of conditional means
            weighted_sum += weights[k] * density_x * conditional_mean_y
            total_weight += weights[k] * density_x

        # Calculate the overall expected value of Y given X = x
        expected_values[i] = weighted_sum / total_weight

    return expected_values


def my_expected_value_and_std_of_y_given_x(model, x_values):
    def gaussian(x, mean, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def px_given_k(x, k):
        return gaussian(x, model.means_[k][0], model.covariances_[k][0, 0])

    def ey_given_x_k(x, k):
        return model.means_[k][1] + model.covariances_[k][0, 1] / model.covariances_[k][0, 0] * (x - model.means_[k][0])

    def vary_given_x_k(x, k):
        return model.covariances_[k][1, 1] - model.covariances_[k][0, 1] ** 2 / model.covariances_[k][0, 0]

    px_given_k_terms = [model.weights_[k] * px_given_k(x_values, k) for k in range(model.n_components)]
    partition = sum(px_given_k_terms)
    components = [term / partition for term in px_given_k_terms]

    ey_given_x = sum([components[k] * ey_given_x_k(x_values, k) for k in range(model.n_components)])
    ey2_given_x = sum([components[k] * (ey_given_x_k(x_values, k) ** 2 + vary_given_x_k(x_values, k)) \
                       for k in range(model.n_components)])
    vary_given_x = ey2_given_x - ey_given_x ** 2

    return ey_given_x, np.sqrt(vary_given_x)


def expected_value_and_std_of_y_given_x(model, x_values):
    # Extract parameters from the GaussianMixture model
    weights = model.weights_  # Mixing weights
    means = model.means_  # Means of each component
    covariances = model.covariances_  # Covariances of each component

    # Prepare to store the expected values and variances
    expected_values = np.zeros_like(x_values)
    variances = np.zeros_like(x_values)

    # Iterate over each x value to calculate E[Y | X = x] and Var[Y | X = x]
    for i, x in enumerate(x_values):
        weighted_sum = 0
        variance_sum = 0
        total_weight = 0

        # Compute weights and conditional means and variances for each component
        for k in range(len(weights)):
            mean_x = means[k, 0]
            mean_y = means[k, 1]
            var_x = covariances[k][0, 0]
            cov_xy = covariances[k][0, 1]
            cov_yx = covariances[k][1, 0]
            var_y = covariances[k][1, 1]

            # Compute the density of x under the k-th Gaussian component's marginal
            density_x = (1 / np.sqrt(2 * np.pi * var_x)) * np.exp(-0.5 * ((x - mean_x) ** 2) / var_x)

            # Compute conditional mean E[Y | X = x] for the k-th component
            conditional_mean_y = mean_y + (cov_yx / var_x) * (x - mean_x)
            # Compute conditional variance Var[Y | X = x] for the k-th component
            conditional_variance_y = var_y - (cov_xy * cov_yx / var_x)

            # Weighted sum of conditional means
            weighted_sum += weights[k] * density_x * conditional_mean_y
            # Weighted sum of variances adjusted for the means
            variance_component = weights[k] * density_x * (conditional_variance_y + conditional_mean_y ** 2)
            variance_sum += variance_component
            total_weight += weights[k] * density_x

        # Calculate the overall expected value of Y given X = x
        expected_values[i] = weighted_sum / total_weight
        # Calculate the overall variance of Y given X = x
        variances[i] = (variance_sum / total_weight) - (expected_values[i] ** 2)

    # Standard deviations are the square root of the variances
    std_deviations = np.sqrt(variances)

    return expected_values, std_deviations


parser = argparse.ArgumentParser(description='Process an Excel file.')
parser.add_argument('ExcelFile', metavar='ExcelFile', type=str, help='the path to an Excel file')
parser.add_argument('--noremove_fails', action='store_true', help='Do not remove the failed cells')
args = parser.parse_args()

intensity = 'nucTDPintensity_normalized'
redness = 'LV_TDP_KD'

df = pd.read_excel(args.ExcelFile)
df = df.dropna(subset=[intensity, redness])

# plt.scatter(df[intensity], df[redness])
# plt.show()


# # Assume p1 and p2 are defined as follows:
# p1 = [6.33, 9.16]
# p2 = [8.09, 3.53]
#
# # Calculate the slope and y-intercept of the line
# m = (p2[1] - p1[1]) / (p2[0] - p1[0])
# b = p1[1] - m * p1[0]

min_mScarlett = 8.5
min_nucTDP = 7.5

# Create a boolean mask for points that satisfy at least one threshold
# mask = (df[intensity] >= min_nucTDP) | (df[redness] >= min_mScarlett)
# mask = df[redness] >= m * df[intensity] + b

# Save the points that will be filtered out
# outliers = df[~mask]

# Apply the mask to the DataFrame
# df = df[mask]

centers = np.array([[8.57, 11.04],
                    [12.1, 7.35],
                    [8.6, 7.3],
                    [5.48, 5.16]
                    ])

# gmm = GaussianMixture(n_components=3, means_init=centers)
gmm = GaussianMixture(n_components=4, means_init=centers, random_state=239)
data = df[[intensity, redness]].values
gmm.fit(data)
labels = gmm.predict(data)


centers = gmm.means_

# colormap = plt.get_cmap('viridis', gmm.n_components)
# colors = [colormap(i) for i in range(gmm.n_components)]
colors = ['#AF1D5A', '#688D84', '#47A6FF', '#525252']

# Create a gridspec
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

ax_main = plt.subplot(gs[1, 0])
ax_xDist = plt.subplot(gs[0, 0], sharex=ax_main)
ax_yDist = plt.subplot(gs[1, 1], sharey=ax_main)

cluster_names = ['TDP-43 KD, activated TDP-REG', 'Normal TDP-43, inactive TDP-REG', 'TDP-43 KD, inactive TDP-REG', 'Dead']
# Create a scatter plot in the larger subplot
for i in range(gmm.n_components):
    ax_main.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=cluster_names[i])

# Print the proportions of each Gaussian
for i, weight in enumerate(gmm.weights_):
    print(f"Percentage of Gaussian {i + 1}: {weight * 100:.2f}%")

num_points_per_class = [sum(labels == i) for i in range(gmm.n_components)]
for i, num_points in enumerate(num_points_per_class):
    print(f"percentage of points in cluster \"{cluster_names[i]}\": {num_points / len(labels) * 100:.2f}%")

# Plot the centers of the Gaussians
ax_main.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.7)

# Plot ovals corresponding to one and two standard deviations
for i in range(gmm.n_components):
    v, w = np.linalg.eigh(gmm.covariances_[i][:2, :2])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    for j in [1, 2]:
        ell = patches.Ellipse(gmm.means_[i, :2], j * v[0], j * v[1], angle=180 + angle, color='black')
        ell.set_clip_box(ax_main.bbox)
        ell.set_alpha(0.1)
        ax_main.add_artist(ell)

# add the outliers
# ax_main.scatter(outliers[intensity], outliers[redness], c='black', s=20, marker='x', label='Dead')

# plot the line
# x = np.linspace(4.75, 8.75, 100)
# y = m * x + b
# ax_main.plot(x, y, color='black', label='Dead cutoff')

# x = [3.35, min_nucTDP, min_nucTDP]
# y = [min_mScarlett, min_mScarlett, 1.8]
# ax_main.plot(x, y, color='black', label='Dead cutoff')

for ind, ax in enumerate([ax_xDist, ax_yDist]):
    # Create a new figure
    # fig, ax = plt.subplots()
    # plt.title(f'Axis {ind+1}')

    # Define the bins
    minx = data[:, ind].min()
    maxx = data[:, ind].max()
    numbins = 30
    binwidth = (maxx - minx) / numbins
    bins = np.linspace(minx, maxx, numbins)

    orientation = 'vertical' if ax == ax_xDist else 'horizontal'

    # Separate data into groups based on labels
    for i in range(gmm.n_components):
        group_data = data[labels == i, ind]
        ax.hist(group_data, bins=bins, alpha=0.5, label=f'Component {i + 1}', color=colors[i],
                orientation=orientation)

    # Plot the PDF of each Gaussian along the x-axis
    for i in range(gmm.n_components):
        mean = gmm.means_[i, ind]
        var = gmm.covariances_[i, ind, ind]
        weight = gmm.weights_[i]
        x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
        y = stats.norm.pdf(x, mean, np.sqrt(var))
        # y = weight*stats.norm.pdf(x, mean, np.sqrt(var))
        y *= binwidth * len(data[labels == i])
        if orientation == 'vertical':
            ax.plot(x, y, color=colors[i], linestyle='--')
            ax.set_ylabel('Count')
        else:
            ax.plot(y, x, color=colors[i], linestyle='--')
            ax.set_xlabel('Count')

    # Add a legend
    # ax.legend()

if not args.noremove_fails:
    # remove the cluster without activations
    # Identify the cluster to remove
    clusters_to_remove = [2, 3]  # for example, to remove the third cluster

    # Remove the elements of the array corresponding to clusters_to_remove
    gmm.weights_ = np.delete(gmm.weights_, clusters_to_remove)
    gmm.means_ = np.delete(gmm.means_, clusters_to_remove, axis=0)
    gmm.covariances_ = np.delete(gmm.covariances_, clusters_to_remove, axis=0)




    # Renormalize the weights
    gmm.weights_ /= gmm.weights_.sum()

    # Update the number of components
    gmm.n_components -= len(clusters_to_remove)

xx = np.linspace(6.9, 13.5, 100)
# yy, std = expected_value_and_std_of_y_given_x(gmm, xx)
my_yy, my_std = my_expected_value_and_std_of_y_given_x(gmm, xx)

# Plot the new expected value curve
# ax_main.plot(xx, yy, color='blue', label=f'Expected redness given intensity ({gmm.n_components} clusters)')
ax_main.plot(xx, my_yy, color='red',
             label=f'Expected mScarlett intensity given the intensity of nuclear TDP-43')
# ax_main.fill_between(xx, yy - std, yy + std, color='blue', alpha=0.2)
ax_main.fill_between(xx, my_yy - my_std, my_yy + my_std, color='red', alpha=0.2)

# xx = np.linspace(6.9, 13.5, 100)
# yy, std = expected_value_and_std_of_y_given_x(gmm, xx)
# ax_main.plot(xx, yy, color='red', label='Expected redness given intensity')
# # plot the standard deviations as a shaded region
# ax_main.fill_between(xx, yy - std, yy + std, color='red', alpha=0.2)

# Also plot the standard deviation on the axx[1]
# ax = axx[1]
# ax.plot(xx, std, color='red', label='Standard deviation of Y given X')
# ax.set_ylim(0, 2.5)

ax_main.set_xlabel('Nuclear TDP-43 Intensity (log-transformed)')
ax_main.set_ylabel('mScarlett Intensity (log-transformed)')

legend = ax_main.legend()
legend.set_draggable(True)




# Hide the labels of the histograms
plt.setp(ax_xDist.get_xticklabels(), visible=False)
plt.setp(ax_yDist.get_yticklabels(), visible=False)

plt.show()

plt.show()
