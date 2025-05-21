#!/usr/bin/env python3
"""
Entry point for firmware: load configuration and start the scheduler.
"""

from miniflight.utils import load_config, log
from miniflight.scheduler import Scheduler
from miniflight.hal import HAL
from targets import get_target

def main():
    config = load_config()
    log("Configuration loaded")
    # select target
    target_name = config.get("target", "default")
    try:
        target_cfg = get_target(target_name)
        log(f"Selected target: {target_name}")
    except KeyError:
        log(f"Target not found: {target_name}")
        return
    # initialize HAL with target config
    hal = HAL(target_cfg)
    log("HAL initialized")
    scheduler = Scheduler(config)
    log("Starting scheduler")
    scheduler.run()

if __name__ == "__main__":
    main() 