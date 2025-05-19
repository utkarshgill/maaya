#!/usr/bin/env python3
"""imu_stream.py – Stream live accelerometer & gyro values from a Betaflight/INAV board.

Pure-Python, no external packages.

Usage:
    python3 imu_stream.py [/dev/tty.usbmodemXXXX] [baud]

Output columns:
    time  ax ay az  gx gy gz   (raw ADC counts)
"""
import glob
import os
import select
import struct
import sys
import termios
import time

# -----------------------------------------------------------------------------
# MSP helpers
# -----------------------------------------------------------------------------

CMD_RAW_IMU = 102  # MSP_RAW_IMU


def msp_request(cmd: int) -> bytes:
    """Build minimal MSPv1 request frame."""
    size = 0
    checksum = size ^ cmd
    return b"$M<" + bytes([size, cmd, checksum])

RAW_IMU_REQ = msp_request(CMD_RAW_IMU)

# -----------------------------------------------------------------------------
# Serial helpers
# -----------------------------------------------------------------------------

def open_serial(device: str, baud: int = 115200) -> int:
    fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0  # iflag
    attrs[1] = 0  # oflag
    attrs[2] = termios.CREAD | termios.CLOCAL | termios.CS8
    attrs[3] = 0  # lflag
    baud_const = getattr(termios, f"B{baud}", None)
    if baud_const is None:
        raise ValueError(f"Unsupported baud {baud}")
    if hasattr(termios, "cfsetispeed"):
        termios.cfsetispeed(attrs, baud_const)
        termios.cfsetospeed(attrs, baud_const)
    else:
        attrs[4] = baud_const  # ispeed
        attrs[5] = baud_const  # ospeed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


def auto_device() -> str:
    patterns = [
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/tty.SLAB_USBtoUART*",
        "/dev/ttyUSB*",
    ]
    for pat in patterns:
        lst = glob.glob(pat)
        if lst:
            return lst[0]
    raise FileNotFoundError("No serial device found; specify path explicitly.")

# -----------------------------------------------------------------------------
# Frame parser
# -----------------------------------------------------------------------------

def parse_frames(buf: bytearray):
    i = 0
    while True:
        start = buf.find(b"$M>", i)
        if start == -1 or start + 5 > len(buf):
            break
        size = buf[start + 3]
        cmd = buf[start + 4]
        end = start + 5 + size
        if end >= len(buf):
            break
        payload = bytes(buf[start + 5 : end])
        crc = buf[end]
        calc = size ^ cmd
        for b in payload:
            calc ^= b
        if calc == crc:
            yield cmd, payload
            i = end + 1
        else:
            i = start + 1  # skip bad header
    del buf[:i]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else auto_device()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    fd = open_serial(device, baud)
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    buf = bytearray()
    next_req = 0.0
    print("time\tax\tay\taz\tgx\tgy\tgz")
    try:
        while True:
            now = time.time()
            if now >= next_req:
                os.write(fd, RAW_IMU_REQ)
                next_req = now + 0.02  # 50 Hz
            if poller.poll(10):  # 10 ms
                try:
                    chunk = os.read(fd, 512)
                    buf.extend(chunk)
                except BlockingIOError:
                    pass
            for cmd, payload in list(parse_frames(buf)):
                if cmd != CMD_RAW_IMU or len(payload) < 12:
                    continue
                ax, ay, az, gx, gy, gz = struct.unpack_from("<hhhhhh", payload)
                print(f"{now:.3f}\t{ax}\t{ay}\t{az}\t{gx}\t{gy}\t{gz}")
    except KeyboardInterrupt:
        pass
    finally:
        os.close(fd)


if __name__ == "__main__":
    main() 