<div align="center">
<picture>
   <img src="https://github.com/user-attachments/assets/e083cd13-de4e-4212-91ec-881500eb5bd8" alt="miniflight logo" width="30%"/>
</picture>

miniflight is a minimal flight control firmware.

</div>

---

This may not be the best flight stack, but it is a flight stack.

Open source flight controllers are bloated, complex, and nearly impossible to debug or extend. Due to its extreme simplicity, miniflight aims to be the easiest controller to add targets to, with support for both config and simulation.

![Screen Recording 2025-05-19 at 9 16 49 AM (6)](https://github.com/user-attachments/assets/62436609-37f9-44b1-bcea-3ab8a77b1491)

## Getting Started

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/utkarshgill/miniflight.git
   cd miniflight
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
