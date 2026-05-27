import torch
import math

# Constants
g = torch.tensor(9.81)
k = torch.tensor(0.1)  
target_x = torch.tensor(15.0)
target_y = torch.tensor(2.0)

# Initial Guesses
v0_mag = torch.tensor(20.0, requires_grad=True)
theta = torch.tensor([45.0 * math.pi / 180.0], requires_grad=True)

# Adam optimizer and Scheduler
optimizer = torch.optim.Adam([v0_mag, theta], lr=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

best_loss = float('inf')
best_theta = 0.0
best_v0 = 0.0

# --- THE COURANT CONDITION PARAMETER ---
# We want the ball to move a maximum of 1cm (0.01m) per frame to avoid skipping the target
max_dx = torch.tensor(0.01) 

for step in range(300):
    optimizer.zero_grad()
    
    x, y = torch.tensor(0.0), torch.tensor(0.0)
    vx = v0_mag * torch.cos(theta)
    vy = v0_mag * torch.sin(theta)
    
    min_loss = torch.tensor(float('inf'))
    
    # Run simulation for more steps (since dt is now much smaller)
    for _ in range(2000):
        # Current total velocity
        v = torch.sqrt(vx**2 + vy**2)
        
        # --- ADAPTIVE TIME STEP (CFL) ---
        # dt = distance / velocity. (Adding 1e-6 prevents division by zero if v=0)
        dt = max_dx / (v + 1e-6)
        
        ax = -k * v * vx
        ay = -g - k * v * vy
        
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        
        # Calculate distance to target
        current_distance_sq = (x - target_x)**2 + (y - target_y)**2
        min_loss = torch.min(min_loss, current_distance_sq)
        
        # Stop this specific throw if it hits the ground (to save calculation time)
        if y < 0 and _ > 10:
            break

    current_loss_val = min_loss.item()

    if current_loss_val < best_loss:
        best_loss = current_loss_val
        best_theta = theta.item()
        best_v0 = v0_mag.item()

    # The Emergency Brake (Stops the outer loop)
    # 1e-8 loss means the distance squared is 0.00000001, which is exactly 0.1mm away
    if current_loss_val < 1e-8: 
        print(f"Target hit perfectly at step {step}! Stopping early.")
        break

    min_loss.backward()
    optimizer.step()
    scheduler.step(min_loss.detach())

    if step % 20 == 0:
        print(f"Step {step:3}: Loss = {current_loss_val:.8f}")

print(f"\n--- GLOBAL BEST ---")
print(f"Result: {best_theta*180/math.pi:.2f}°, {best_v0:.2f}m/s")
print(f"Loss: {best_loss:.8f}")
print(f"Actual Distance Missed By: {math.sqrt(best_loss)*1000:.2f} mm")