#!/usr/bin/env python3
"""
Entry point for firmware: load configuration, setup HAL and scheduler, and run control loop.
"""
import json

from common.logger import get_logger
from common.scheduler import Scheduler
from firmware.control import StabilityController
from firmware.hal      import HAL

logger = get_logger("firmware")

def load_config(path: str = "config.json") -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"Config file '{path}' not found, using defaults.")
        return {}


def main():
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")

    # Dispatch to simulator if requested
    target_name = config.get("target", None)
    if target_name == "sim":
        from target.sim import Simulator
        Simulator().run()
        return

    # Real-hardware firmware path
    controller = StabilityController()
    hal = HAL(config, controller)
    logger.info("HAL initialized (HAL)")

    # Setup multi-rate scheduler
    dt = config.get("dt", 0.01)
    scheduler = Scheduler()
    scheduler.add_task(lambda: hal.run_cycle(dt), period=dt)

    logger.info("Starting scheduler loop")
    scheduler.run()

if __name__ == "__main__":
    from target.sim import Simulator
    Simulator().run() 