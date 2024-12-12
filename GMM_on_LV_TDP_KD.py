import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture


def expected_value_and_std_of_y_given_x(model, x_values):
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


# def discr_theshold(weight1, mean1, var1, weight2, mean2, var2):
#     prior_odds = weight1 / weight2
#     likelihood_


# For finding the boundary where the posterior probability of belong to the two components are equal
def find_decision_boundary_x(gmm, x_range):
    boundaries = []
    previous_label = None
    for x in x_range:
        probabilities = [
            gmm.weights_[k] * stats.norm.pdf(x, gmm.means_[k][0], np.sqrt(gmm.covariances_[k][0, 0]))
            for k in range(gmm.n_components)
        ]
        normalized_probs = np.array(probabilities) / sum(probabilities)
        current_label = np.argmax(normalized_probs)
        if previous_label is not None and current_label != previous_label:
            boundaries.append(x)
        previous_label = current_label
    return boundaries

def find_decision_boundary_y(gmm, y_range):
    boundaries = []
    previous_label = None
    for y in y_range:
        probabilities = [
            gmm.weights_[k] * stats.norm.pdf(y, gmm.means_[k][1], np.sqrt(gmm.covariances_[k][1, 1]))
            for k in range(gmm.n_components)
        ]
        normalized_probs = np.array(probabilities) / sum(probabilities)
        current_label = np.argmax(normalized_probs)
        if previous_label is not None and current_label != previous_label:
            boundaries.append(y)
        previous_label = current_label
    return boundaries


parser = argparse.ArgumentParser(description='Process an Excel file.')
parser.add_argument('ExcelFile', metavar='ExcelFile', type=str, help='the path to an Excel file')
parser.add_argument('--noremove_fails', action='store_true', help='Do not remove the failed cells')
# parser.add_argument('--ncomponents', type=int, default=2, help='Number of components in the Gaussian Mixture Model')
args = parser.parse_args()


intensity = 'nucTDP_intensity_normalized_log'
redness = 'reporter_intensity_normalized_log'

df = pd.read_excel(args.ExcelFile)
df = df.dropna(subset=[intensity, redness])

# Visualize the data as it is
plt.scatter(df[intensity], df[redness], alpha=0.1)
plt.title('NucTDP Intensity vs Reporter Intensity, All nuclei')
plt.xlabel('NucTDP Intensity (Log)')
plt.ylabel('Reporter Intensity (Log)')

plt.figure()

# Apply a mask to filter out those with extreme low TDP/mScarlet intensity, usually dead cells
p1 = [11.64, 5.05]
p2 = [10.06, 9.90]
m = (p2[1] - p1[1]) / (p2[0] - p1[0])
b = p1[1] - m * p1[0]
mask = df[redness] >= m * df[intensity] + b

# Alternative masking based on thresholds, now disabled
# min_mScarlett = 8.5
# min_nucTDP = 7.5
# mask = (df[intensity] >= min_nucTDP) | (df[redness] >= min_mScarlett)

# Save the points that will be filtered out
outliers = df[~mask]

# Apply the mask to the DataFrame
df = df[mask]

# How many components to use?
n_components = list(range(1, 11))
aics = []
bics = []
data = df[[intensity, redness]].values
# Fit GMMs and calculate AIC and BIC
for n in n_components:
    gmm = GaussianMixture(n_components=n)
    gmm.fit(data)
    aics.append(gmm.aic(data))
    bics.append(gmm.bic(data))

# Plotting AIC and BIC
plt.plot(n_components, aics, label='AIC', marker='o')
plt.plot(n_components, bics, label='BIC', marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Scores')
plt.title('GMM - AIC and BIC Scores by Number of Components')
plt.legend()
plt.grid(True)
plt.figure()

# Alternatively assign centers for initialization, now disabled
# centers = np.array([[8.57, 11.04],
#                     [12.1, 7.35],
#                     [8.6, 7.3],
#                     [5.48, 5.16]
#                     ])

gmm = GaussianMixture(n_components=2,
                      # means_init=centers,
                      random_state = 239)
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


# Grid for decision boundary visualization
x_min, x_max = df[intensity].min(), df[intensity].max()
y_min, y_max = df[redness].min(), df[redness].max()
x_range = np.linspace(x_min, x_max, 1000)
y_range = np.linspace(y_min, y_max, 1000)

# Find decision boundaries on X-axis and Y-axis
boundaries_x = find_decision_boundary_x(gmm, x_range)
boundaries_y = find_decision_boundary_y(gmm, y_range)
print(f"Decision boundary on X-axis: {boundaries_x}")
print(f"Decision boundary on Y-axis: {boundaries_y}")


cluster_names = ['TDP-43 KD, activated TDP-REG', 'Normal TDP-43, inactive TDP-REG', 'TDP-43 KD, inactive TDP-REG', 'Dead']
# Create a scatter plot in the larger subplot
for i in range(gmm.n_components):
    ax_main.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=cluster_names[i])

# Highlight decision boundaries on X-axis and Y-axis
for boundary in boundaries_x:
    ax_main.axvline(boundary, color='#AF1D5A', linestyle='--', label=f'Activation Boundary at X={boundary:.2f}')

# sometimes two decision boundaries will be ploted because there are 2 components
# for boundary in boundaries_y:
#    ax_main.axhline(boundary, color='blue', linestyle='--', label=f'Decision Boundary at Y={boundary:.2f}')

# Print the proportions of each Gaussian
for i, weight in enumerate(gmm.weights_):
    print(f"Percentage of Gaussian {i + 1}: {weight * 100:.2f}%")

num_points_per_class = [sum(labels == i) for i in range(gmm.n_components)]
for i, num_points in enumerate(num_points_per_class):
    print(f"percentage of points in cluster \"{cluster_names[i]}\": {num_points / len(labels) * 100:.2f}%")

print(f"Centers: ({centers[0, 0]:.2f}, {centers[0, 1]:.2f}), ({centers[1, 0]:.2f}, {centers[1, 1]:.2f})")

# Plot the centers of the Gaussians
ax_main.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.7)
ax_minx = 10
ax_maxx = 13.5
ax_main.set_xlim(ax_minx, ax_maxx)

# Find the turning points
xx_dense = np.linspace(ax_minx, ax_maxx, 1000)
yy_dense, std_dense = expected_value_and_std_of_y_given_x(gmm, xx_dense)
yy_deriv2 = np.gradient(np.gradient(yy_dense, xx_dense), xx_dense)

# find min and max
min_index = np.argmin(yy_deriv2)
max_index = np.argmax(yy_deriv2)
min_point = (xx_dense[min_index], yy_dense[min_index])
max_point = (xx_dense[max_index], yy_dense[max_index])

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
ax_main.scatter(outliers[intensity], outliers[redness], c='black', s=20, marker='x', label='Dead')

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

# xx = np.linspace(6.9, 13.5, 100)
xx = np.linspace(ax_minx, ax_maxx, 500)
yy, std = expected_value_and_std_of_y_given_x(gmm, xx)

# part of xx between the turning points
xx_segment = xx[np.where((xx >= min_point[0]) & (xx <= max_point[0]))]
yy_segment = yy[np.where((xx >= min_point[0]) & (xx <= max_point[0]))]

# Plot the new expected value curve
# ax_main.plot(xx, yy, color='blue', label=f'Expected redness given intensity ({gmm.n_components} clusters)')
ax_main.plot(xx, yy, color='red',
             label=f'Expected mScarlet intensity given the intensity of nuclear TDP-43')
# ax_main.fill_between(xx, yy - std, yy + std, color='blue', alpha=0.2)
ax_main.fill_between(xx, yy - std, yy + std, color='red', alpha=0.2)

ax_main.plot(xx_segment, yy_segment, color='#ffc75f',
             label=f'Linear increase regime')
# Plot the turning points
ax_main.scatter([min_point[0], max_point[0]], [min_point[1], max_point[1]], c='#ffc75f', s=50, alpha=1, marker='D')
print(f"Turning points: min: ({min_point[0]:.2f}, {min_point[1]:.2f}), max: ({max_point[0]:.2f}, {max_point[1]:.2f})")


# Compute the avg derivative in the linear increase regime
xx_dense = np.linspace(min_point[0], max_point[0], 1000)
yy_dense, std_dense = expected_value_and_std_of_y_given_x(gmm, xx_dense)
yy_deriv = np.gradient(yy_dense, xx_dense)
print(f"Average derivative in the linear increase regime: {float(np.mean(yy_deriv))}")


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
ax_main.set_ylabel('mScarlet Intensity (log-transformed)')

legend = ax_main.legend(loc='lower left')
legend.set_draggable(True)


# Hide the labels of the histograms
plt.setp(ax_xDist.get_xticklabels(), visible=False)
plt.setp(ax_yDist.get_yticklabels(), visible=False)

# plt.figure()
# plt.plot(xx_dense, yy_deriv2)
# plt.xlim(ax_minx, ax_maxx)


plt.show()
