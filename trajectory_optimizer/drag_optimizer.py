import torch
import math

# Constants
g = torch.tensor(9.81)
k = torch.tensor(0.1)  
dt = torch.tensor(0.05) 
target_x = torch.tensor(15.0)
target_y = torch.tensor(2.0)


# Initial Guesses
v0_mag = torch.tensor(20.0, requires_grad=True)
theta = torch.tensor([45.0 * math.pi / 180.0], requires_grad=True)

# Adam optimizer controlling BOTH variables
optimizer = torch.optim.Adam([v0_mag, theta], lr=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

best_loss = float('inf')
best_theta = 0.0
best_v0 = 0.0

for step in range(100):
    optimizer.zero_grad()
    
    x, y = torch.tensor(0.0), torch.tensor(0.0)
    vx = v0_mag * torch.cos(theta)
    vy = v0_mag * torch.sin(theta)
    
    # We will track the minimum distance squared to the target
    # Start it at infinity
    min_loss = torch.tensor(float('inf'))
    
    # Run simulation for 100 steps (5 seconds)
    for _ in range(100):
        v = torch.sqrt(vx**2 + vy**2)
        ax = -k * v * vx
        ay = -g - k * v * vy
        
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        
        # Calculate distance to target AT THIS EXACT FRAME
        # D^2 = (x - target_x)^2 + (y - target_y)^2
        current_distance_sq = (x - target_x)**2 + (y - target_y)**2
        
        # If this is the closest we've been, save it!
        # torch.min() is fully differentiable and safe for AI
        min_loss = torch.min(min_loss, current_distance_sq)

        current_loss_val = min_loss.item()

        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_theta = theta.item()
            best_v0 = v0_mag.item()

        if current_loss_val < 1e-6: # 1e-6 is 0.000001
            print(f"Target hit perfectly at step {step}! Stopping early.")
            break

    # Backpropagate based on the closest point in the trajectory
    min_loss.backward()
    optimizer.step()
    scheduler.step(min_loss)
    if step % 1 == 0:
        print(f"Step {step:3}: Angle = {theta.item()*180/math.pi:.2f}°, Velocity = {v0_mag.item():.2f}m/s, Loss = {min_loss.item():.8f}")

print(f"\n--- GLOBAL BEST RESULTS AT Step {step:3} ---")
print(f"Best Angle: {best_theta * 180 / math.pi:.2f}°")
print(f"Best Velocity: {best_v0:.2f}m/s")
print(f"Best Loss: {best_loss:.8f}")