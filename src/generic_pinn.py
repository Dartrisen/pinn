import torch

# =============================================================================
# Generic PINN Framework
# =============================================================================


class GenericPINNSolver:
    def __init__(self,
                 model,  # The neural network model (torch.nn.Module)
                 pde_residual_func,  # Function to compute PDE residual loss
                 bc_loss_func,  # Function to compute boundary condition loss
                 ic_loss_func=None,  # Optional: function to compute initial condition loss
                 collocation_points=None,  # Tensor of collocation points for PDE residual
                 bc_points=None,  # Tensor of boundary points (or a list/dict as needed)
                 ic_points=None,  # Tensor of initial condition points (if needed)
                 device=None):
        """
        Generic PINN solver.

        Parameters:
          model: PyTorch neural network model.
          pde_residual_func: function(model, collocation_points) -> residual tensor.
          bc_loss_func: function(model, bc_points) -> scalar boundary loss.
          ic_loss_func: (optional) function(model, ic_points) -> scalar initial loss.
          collocation_points: Tensor of points for PDE residual evaluation.
          bc_points: Tensor (or dictionary) of points for boundary conditions.
          ic_points: Tensor of points for initial conditions.
          device: torch.device to run computations.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.pde_residual_func = pde_residual_func
        self.bc_loss_func = bc_loss_func
        self.ic_loss_func = ic_loss_func
        self.collocation_points = collocation_points.to(self.device) if collocation_points is not None else None
        self.bc_points = bc_points.to(self.device) if bc_points is not None else None
        self.ic_points = ic_points.to(self.device) if ic_points is not None else None

    def train(self,
              epochs,
              optimizer,
              loss_weights,  # Dictionary for weights e.g. {'pde':1.0, 'bc':10.0, 'ic':1.0}
              log_interval=100):
        """
        Train the PINN.

        Returns:
          loss_history: list of total loss values logged.
        """
        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_pde = self.pde_residual_func(self.model, self.collocation_points)
            loss_bc = self.bc_loss_func(self.model, self.bc_points)
            loss = loss_weights.get('pde', 1.0) * loss_pde + loss_weights.get('bc', 1.0) * loss_bc

            if self.ic_loss_func is not None and self.ic_points is not None:
                loss_ic = self.ic_loss_func(self.model, self.ic_points)
                loss += loss_weights.get('ic', 1.0) * loss_ic
            else:
                loss_ic = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            if epoch % log_interval == 0:
                loss_history.append(loss.item())
                print(f"Epoch {epoch:04d}: Total Loss = {loss.item():.6f}, PDE Loss = {loss_pde.item():.6f}, BC Loss = {loss_bc.item():.6f}, IC Loss = {loss_ic.item():.6f}")
        return loss_history

    def predict(self, x):
        """Return the model prediction at x (a NumPy array or torch.Tensor)."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if len(x_tensor.shape) == 1:
                x_tensor = x_tensor.unsqueeze(1)
            prediction = self.model(x_tensor)
        return prediction.cpu().numpy()
