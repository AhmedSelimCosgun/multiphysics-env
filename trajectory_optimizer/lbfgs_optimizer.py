import torch
import math

# Target coordinates (30m away, 5m high)
v0 = torch.tensor(20.0)
g = torch.tensor(9.81)
target_x = torch.tensor(30.0)
target_y = torch.tensor(5.0)

# Start with a guess of 10 degrees
theta = torch.tensor([10.0 * math.pi / 180.0], requires_grad=True)

# L-BFGS usually needs a learning rate of 1.0 (it manages its own step size)
optimizer = torch.optim.LBFGS([theta], lr=1)

def closure():
    optimizer.zero_grad()
    # Physical equation for Y height at a specific X distance
    current_y = (target_x * torch.tan(theta)) - ((g * target_x**2) / (2 * v0**2 * torch.cos(theta)**2))
    loss = (current_y - target_y)**2
    loss.backward()
    return loss

print("Running L-BFGS...")
for i in range(10):  # It should solve this VERY fast
    optimizer.step(closure)
    
    # Check the result
    final_loss = closure()
    print(f"Step {i}: Angle = {theta.item()*180/math.pi:.4f}°, Loss = {final_loss.item():.6f}")

print(f"\nOptimization Complete. Final Angle: {theta.item()*180/math.pi:.2f}°")