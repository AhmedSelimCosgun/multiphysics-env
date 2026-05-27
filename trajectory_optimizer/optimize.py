import torch
import math
import matplotlib.pyplot as plt

# 1. Physical Parameters (Using PyTorch Tensors instead of standard numbers)
v0 = torch.tensor(20.0)
g = torch.tensor(9.81)
target_x = torch.tensor(35.0) # We want the ball to land exactly at 35m

# 2. The Variable we want the "AI" to learn
# We start with a bad guess: 10 degrees. 
# requires_grad=True tells PyTorch: "Track the calculus for this variable!"
theta_guess = torch.tensor([10.0 * math.pi / 180.0], requires_grad=True)

# 3. The Optimizer (This is the engine of all Neural Networks)
# lr is the "learning rate" (how big of a step it takes when adjusting)
optimizer = torch.optim.Adam([theta_guess], lr=0.05)

# 4. The Training Loop
print("Starting optimization...")
for step in range(200):
    optimizer.zero_grad() # Clear old math
    
    current_x = v0**2 * torch.sin(2*theta_guess) / 2 
    
    # --- THE AI PART: THE LOSS FUNCTION ---
    # Calculate the Mean Squared Error (MSE) between where the ball landed and the target.
    # It's just (current - target) squared.
    loss = (current_x - target_x) ** 2 
    
    # Calculate the derivatives automatically!
    loss.backward()
    
    # Adjust the angle based on the derivatives
    optimizer.step()
    
    # Print progress every 20 steps
    if step % 20 == 0:
        angle_deg = theta_guess.item() * 180 / math.pi
        print(f"Step {step:3}: Angle = {angle_deg:.2f}°, Landed at = {current_x.item():.2f}m, Loss = {loss.item():.2f}")

print(f"\nFinal Optimized Angle: {theta_guess.item() * 180 / math.pi:.2f}°")