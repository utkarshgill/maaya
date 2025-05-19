# maaya
Simple physics engine for simulating rigid body dynamics.

![Screen+Recording+2025-05-19+at+9 16 49 AM](https://github.com/user-attachments/assets/9e3d5408-8624-46a4-9b10-c346c3c105fb)


## Getting Started

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/<your-username>/maaya.git
   cd maaya
   ```

2. Install dependencies & create environment:
   ```bash
   bash setup.sh
   conda activate pybullet_env
   ```

3. Run the quadrotor simulation:
   ```bash
   python examples/quad_hover_env.py
   ```

Use PS5 DualSense or keyboard:

- W/S or left stick vertical: throttle up/down
- Arrow keys or right stick: pitch/roll
- A/D or left stick horizontal: yaw
- X/Cross or Spacebar: pick/drop

## More examples
A quadrotor learning to stabilize itself.

![Screen-Recording-2024-07-26-at-4 11 51 PM](https://github.com/user-attachments/assets/0e245827-e067-4a7f-a535-fe2fb6ce15eb)
