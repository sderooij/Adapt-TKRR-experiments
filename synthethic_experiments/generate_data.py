import numpy as np
from scipy.stats import multivariate_normal

# Helper function to generate positive data from Gaussian mixture
def generate_positive_data(centers, cov, num_samples):
    num_samples_per_component = num_samples // len(centers)
    rem_samples = num_samples % len(centers)
    positive_data = []
    for i, center in enumerate(centers):
        if rem_samples > 0 and i == 0:
            datasize = num_samples_per_component + 1
        else:
            datasize = num_samples_per_component
        data = multivariate_normal.rvs(mean=center, cov=cov, size=datasize)
        positive_data.append(data)
    return positive_data

# Helper function to generate negative data uniformly outside the positive area
def generate_negative_data(bounds, exclusion_centers, exclusion_radius, num_samples):
    negative_data = []
    while len(negative_data) < num_samples:
        sample = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_samples, 2))
        distances = [np.linalg.norm(sample - np.array(center), axis=1) for center in exclusion_centers]
        outside_exclusion = np.all(np.column_stack(distances) > exclusion_radius, axis=1)
        negative_data.extend(sample[outside_exclusion])
    return np.array(negative_data[:num_samples])
