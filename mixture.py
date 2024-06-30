import argparse

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
args = parser.parse_args()

intensity = 'nucTDPintensity_normalized'
redness = 'LV_TDP_KD'

df = pd.read_excel(args.ExcelFile)
df = df.dropna(subset=[intensity, redness])

# Assume p1 and p2 are defined as follows:
p1 = [6.33, 9.16]
p2 = [8.09, 3.53]

# Calculate the slope and y-intercept of the line
m = (p2[1] - p1[1]) / (p2[0] - p1[0])
b = p1[1] - m * p1[0]

# Create a boolean mask for points above the line
mask = df[redness] >= m * df[intensity] + b

# Save the points that will be filtered out
outliers = df[~mask]

# Apply the mask to the DataFrame
df = df[mask]


# plt.scatter(df[intensity], df[redness])
# plt.show()

centers = np.array([[8.57, 11.04],
                    [12.1, 7.35],
                    [8.6,  7.3]])

gmm = GaussianMixture(n_components=3, means_init=centers)
data = df[[intensity, redness]].values
gmm.fit(data)
labels = gmm.predict(data)

# Print the proportions of each Gaussian
for i, weight in enumerate(gmm.weights_):
    print(f"Proportion of Gaussian {i+1}: {weight}")

# Create a new figure
fig, axx = plt.subplots(2, 1, sharex=True)

ax = axx[0]
# Plot the classified data
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')

# Plot the centers of the Gaussians
centers = gmm.means_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.7)

# Plot ovals corresponding to one and two standard deviations
for i in range(gmm.n_components):
    v, w = np.linalg.eigh(gmm.covariances_[i][:2, :2])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    for j in [1, 2]:
        ell = patches.Ellipse(gmm.means_[i, :2], j * v[0], j * v[1], angle=180 + angle, color='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.1)
        ax.add_artist(ell)

# add the outliers
ax.scatter(outliers[intensity], outliers[redness], c='black', s=10, marker='x')
# plot the line
x = np.linspace(4.75, 8.75, 100)
y = m * x + b
ax.plot(x, y, color='black', label='Dead cutoff')

xx = np.linspace(6.9, 13.5, 100)
yy, std = expected_value_and_std_of_y_given_x(gmm, xx)
ax.plot(xx, yy, color='red', label='Expected value of Y given X')
# plot the standard deviations as a shaded region
ax.fill_between(xx, yy - std, yy + std, color='red', alpha=0.2)

# Also plot the standard deviation on the axx[1]
ax = axx[1]
ax.plot(xx, std, color='red', label='Standard deviation of Y given X')
ax.set_ylim(0, 2.5)

plt.legend()

for ind in [0, 1]:
    # Create a new figure
    fig, ax = plt.subplots()
    plt.title(f'Axis {ind+1}')

    # Define the bins
    minx = data[:, ind].min()
    maxx = data[:, ind].max()
    numbins = 30
    binwidth = (maxx - minx) / numbins
    bins = np.linspace(minx, maxx, numbins)

    colormap = plt.get_cmap('viridis', gmm.n_components)

    # Separate data into groups based on labels
    for i in range(gmm.n_components):
        group_data = data[labels == i, ind]
        ax.hist(group_data, bins=bins, alpha=0.5, label=f'Component {i+1}', color=colormap(i))

    # Plot the PDF of each Gaussian along the x-axis
    for i in range(gmm.n_components):
        mean = gmm.means_[i, ind]
        var = gmm.covariances_[i, ind, ind]
        weight = gmm.weights_[i]
        x = np.linspace(mean - 3*np.sqrt(var), mean + 3*np.sqrt(var), 100)
        y = stats.norm.pdf(x, mean, np.sqrt(var))
        # y = weight*stats.norm.pdf(x, mean, np.sqrt(var))
        y *= binwidth * len(data[labels == i])
        ax.plot(x, y, color=colormap(i), linestyle='--')

    # Add a legend
    ax.legend()

plt.show()
