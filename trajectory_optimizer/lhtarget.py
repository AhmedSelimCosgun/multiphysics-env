import torch
import math

v0 = torch.tensor(20.0)
g = torch.tensor(9.81)

target_x = torch.tensor(30.0) 
target_y = torch.tensor(5.0)

theta_guess = torch.tensor([10.0 * math.pi / 180.0], requires_grad=True)
optimizer = torch.optim.Adam([theta_guess], lr=0.05)

print("Starting optimization for target (30, 5)...")
for step in range(500):
    optimizer.zero_grad() 
    
    # Calculate the Y height exactly when the ball reaches target_x
    # Equation: y = x*tan(theta) - (g*x^2) / (2*v0^2*cos^2(theta))
    current_y = (target_x * torch.tan(theta_guess)) - ((g * target_x**2) / (2 * v0**2 * torch.cos(theta_guess)**2))
    
    # LOSS: How far is our current Y from the target Y?
    loss = (current_y - target_y) ** 2 
    
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        angle_deg = theta_guess.item() * 180 / math.pi
        print(f"Step {step:3}: Angle = {angle_deg:.2f}°, Y Height = {current_y.item():.2f}m, Loss = {loss.item():.2f}")

print(f"\nFinal Optimized Angle: {theta_guess.item() * 180 / math.pi:.2f}°")