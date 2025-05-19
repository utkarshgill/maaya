#!/usr/bin/env python3
"""msp_sniffer.py – Minimal, dependency-free MSP sniffer for Betaflight/INAV boards.

Usage (macOS / Linux):
    python3 msp_sniffer.py [/dev/tty.usbmodemXXXX] [baud]

If no serial device is supplied, the script will try to auto-detect the first
USB serial device that looks like a flight-controller.

The script:
  • Configures the TTY for 115200 8N1 (or user-supplied baud).
  • Periodically sends MSP_IDENT (100) and MSP_STATUS (101) requests.
  • Prints every valid MSP frame coming FROM the board (direction character '>').

Everything is implemented with the Python standard library only—no pyserial.
"""

import errno
import glob
import os
import select
import struct
import sys
import termios
import time
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# MSP helpers
# -----------------------------------------------------------------------------

CMD_NAMES: Dict[int, str] = {
    100: "MSP_IDENT",
    101: "MSP_STATUS",
    102: "MSP_RAW_IMU",
    103: "MSP_SERVO",
    104: "MSP_MOTOR",
    105: "MSP_RC",
    106: "MSP_RAW_GPS",
    107: "MSP_COMP_GPS",
    108: "MSP_ATTITUDE",
}

def msp_request(cmd: int, payload: bytes = b"") -> bytes:
    """Build an MSP frame (direction '<') to request data from the FC."""
    size = len(payload)
    checksum = size ^ cmd
    for b in payload:
        checksum ^= b
    return b"$M<" + bytes([size, cmd]) + payload + bytes([checksum])

# -----------------------------------------------------------------------------
# Serial helpers (POSIX only)
# -----------------------------------------------------------------------------

def open_serial(device: str, baud: int = 115200) -> int:
    """Open and configure a serial port, return its file-descriptor."""
    fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

    attrs = termios.tcgetattr(fd)

    # Input flags – turn off everything we can to get raw bytes.
    attrs[0] = 0
    # Output flags.
    attrs[1] = 0
    # Control flags.
    attrs[2] = termios.CREAD | termios.CLOCAL | termios.CS8  # 8 data bits.
    # Local flags – turn off canonical mode, echo, etc.
    attrs[3] = 0

    # Baud rate.
    baud_const = getattr(termios, f"B{baud}", None)
    if baud_const is None:
        raise ValueError(f"Unsupported baud rate {baud}")

    # On some macOS/Python builds, cfsetispeed/cfsetospeed are unavailable.
    if hasattr(termios, "cfsetispeed"):
        termios.cfsetispeed(attrs, baud_const)
        termios.cfsetospeed(attrs, baud_const)
    else:
        # attrs[4] is ISPEED, attrs[5] is OSPEED in the attr list.
        attrs[4] = baud_const
        attrs[5] = baud_const

    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd

# -----------------------------------------------------------------------------
# Frame parser
# -----------------------------------------------------------------------------

def parse_msp_frames(buf: bytearray) -> Tuple[List[Tuple[int, bytes]], bytearray]:
    """Extract complete MSP frames from *buf*.

    Returns (frames, remainder) where frames is a list of (cmd, payload).
    The remainder is any trailing bytes that did not form a full frame.
    """
    frames: List[Tuple[int, bytes]] = []
    i = 0
    while i < len(buf):
        start = buf.find(b"$M>", i)
        if start == -1:
            break  # No starting sequence.
        if start + 5 > len(buf):
            break  # Not enough bytes for size+cmd yet.
        size = buf[start + 3]
        cmd = buf[start + 4]
        frame_end = start + 5 + size  # excluding checksum.
        if frame_end >= len(buf):
            break  # Payload not fully received yet.
        payload = bytes(buf[start + 5 : frame_end])
        checksum = buf[frame_end]
        calc = size ^ cmd
        for b in payload:
            calc ^= b
        if calc == checksum:
            frames.append((cmd, payload))
            i = frame_end + 1
        else:
            # Bad checksum – skip this '$' and continue searching.
            i = start + 1
    remainder = buf[i:]
    return frames, bytearray(remainder)

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def auto_detect_device() -> str:
    """Return the first tty.* device that looks like a USB serial port."""
    patterns = [
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/tty.SLAB_USBtoUART*",
        "/dev/ttyUSB*",  # Linux FTDI style.
    ]
    for pattern in patterns:
        devices = sorted(glob.glob(pattern))
        if devices:
            return devices[0]
    raise FileNotFoundError("No USB serial device found. Specify one explicitly.")

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        device = auto_detect_device()

    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    print(f"Opening {device} at {baud} baud …")
    try:
        fd = open_serial(device, baud)
    except PermissionError:
        sys.exit("Permission denied opening serial port. Try again with sudo or add user to dialout group.")
    except Exception as e:
        sys.exit(str(e))

    ident_req = msp_request(100)  # MSP_IDENT
    status_req = msp_request(101)  # MSP_STATUS

    # Non-blocking read loop.
    poller = select.poll()
    poller.register(fd, select.POLLIN)

    buf = bytearray()
    next_query = 0.0
    print("Press Ctrl-C to quit.")

    try:
        while True:
            now = time.time()
            if now >= next_query:
                # Send a pair of simple requests every second.
                os.write(fd, ident_req)
                os.write(fd, status_req)
                next_query = now + 1.0

            # Wait up to 100 ms for data.
            events = poller.poll(100)
            if events:
                try:
                    chunk = os.read(fd, 1024)
                    buf.extend(chunk)
                except OSError as e:
                    if e.errno != errno.EAGAIN:
                        raise
            frames, buf = parse_msp_frames(buf)
            for cmd, payload in frames:
                name = CMD_NAMES.get(cmd, f"CMD_{cmd}")
                print(f"{time.strftime('%H:%M:%S')} | {name} ({cmd}) | {len(payload)} bytes | {payload.hex()}")
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main() 