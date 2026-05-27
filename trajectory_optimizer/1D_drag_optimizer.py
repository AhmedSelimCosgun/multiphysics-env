import torch
import math

# Strict 64-bit precision for the deep dive
dtype = torch.float64

g = torch.tensor(9.81, dtype=dtype)
k = torch.tensor(0.1, dtype=dtype)  
dt = torch.tensor(0.01, dtype=dtype) 
total_sim_time = 5.0             
sim_steps = int(total_sim_time / dt.item()) 

target_x = torch.tensor(15.0, dtype=dtype)
target_y = torch.tensor(2.0, dtype=dtype)

target_angle_deg = 38.0 
theta = torch.tensor([target_angle_deg * math.pi / 180.0], dtype=dtype) 

v0_mag = torch.tensor(20.0, dtype=dtype, requires_grad=True)

# ---------------------------------------------------------
# PHASE 1: ADAM (The Ballistic Descent)
# ---------------------------------------------------------
print(f"--- PHASE 1: ADAM OPTIMIZATION ---")
optimizer_adam = torch.optim.Adam([v0_mag], lr=0.3)

for step in range(40):
    optimizer_adam.zero_grad()
    
    x, y = torch.tensor(0.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    vx = v0_mag * torch.cos(theta)
    vy = v0_mag * torch.sin(theta)
    
    min_segment_dist_sq = torch.tensor(float('inf'), dtype=dtype)
    
    for _ in range(sim_steps):
        x_old, y_old = x, y
        v = torch.sqrt(vx**2 + vy**2)
        ax = -k * v * vx
        ay = -g - k * v * vy
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        
        seg_x, seg_y = x - x_old, y - y_old
        seg_len_sq = seg_x**2 + seg_y**2 + 1e-20
        target_vec_x, target_vec_y = target_x - x_old, target_y - y_old
        
        t = torch.clamp((target_vec_x * seg_x + target_vec_y * seg_y) / seg_len_sq, 0.0, 1.0)
        closest_x, closest_y = x_old + t * seg_x, y_old + t * seg_y
        
        dist_sq = (closest_x - target_x)**2 + (closest_y - target_y)**2
        min_segment_dist_sq = torch.where(dist_sq < min_segment_dist_sq, dist_sq, min_segment_dist_sq)

    loss = min_segment_dist_sq
    
    if step % 5 == 0:
        print(f"Adam Step {step:2}: Velocity = {v0_mag.item():.4f}m/s, Loss = {loss.item():.8f}")
        
    loss.backward()
    optimizer_adam.step()

# ---------------------------------------------------------
# PHASE 2: L-BFGS (The Precision Finish)
# ---------------------------------------------------------
print(f"\n--- PHASE 2: L-BFGS OPTIMIZATION ---")
# L-BFGS needs a slightly different setup with a closure function
optimizer_lbfgs = torch.optim.LBFGS([v0_mag], lr=1.0, max_iter=20, tolerance_change=1e-12)

def closure():
    optimizer_lbfgs.zero_grad()
    x, y = torch.tensor(0.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    vx = v0_mag * torch.cos(theta)
    vy = v0_mag * torch.sin(theta)
    
    min_segment_dist_sq = torch.tensor(float('inf'), dtype=dtype)
    
    for _ in range(sim_steps):
        x_old, y_old = x, y
        v = torch.sqrt(vx**2 + vy**2)
        ax = -k * v * vx
        ay = -g - k * v * vy
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
        
        seg_x, seg_y = x - x_old, y - y_old
        seg_len_sq = seg_x**2 + seg_y**2 + 1e-20
        target_vec_x, target_vec_y = target_x - x_old, target_y - y_old
        
        t = torch.clamp((target_vec_x * seg_x + target_vec_y * seg_y) / seg_len_sq, 0.0, 1.0)
        closest_x, closest_y = x_old + t * seg_x, y_old + t * seg_y
        
        dist_sq = (closest_x - target_x)**2 + (closest_y - target_y)**2
        min_segment_dist_sq = torch.where(dist_sq < min_segment_dist_sq, dist_sq, min_segment_dist_sq)

    loss = min_segment_dist_sq
    loss.backward()
    return loss

# Run L-BFGS
for step in range(5):
    loss = optimizer_lbfgs.step(closure)
    real_loss = loss.item()
    print(f"L-BFGS Step {step}: Velocity = {v0_mag.item():.8f}m/s, Loss = {real_loss:.12f}")
    
    if real_loss < 1e-6:
        print(f"\nTarget nailed! Precision achieved.")
        break

print(f"\n--- GLOBAL BEST RESULTS ---")
print(f"Fixed Angle: {target_angle_deg}°")
print(f"Final Velocity: {v0_mag.item():.8f} m/s")
print(f"Final Loss: {loss.item():.12f}")