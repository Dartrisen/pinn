# Define domain limits
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
N_x, N_y = 50, 50  # resolution in each dimension

# Generate a mesh grid
x_vals = np.linspace(x_min, x_max, N_x)
y_vals = np.linspace(y_min, y_max, N_y)
X, Y = np.meshgrid(x_vals, y_vals)
# Flatten the grid into a list of collocation points
collocation_points = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
collocation_tensor = torch.tensor(collocation_points, dtype=torch.float32)
