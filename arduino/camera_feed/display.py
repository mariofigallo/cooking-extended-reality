import serial
import numpy as np
import cv2
import struct

# ── Configuration ─────────────────────────────────────────────────────────────
PORT = '/dev/cu.usbserial-10'       # Windows: e.g. COM5  |  Linux/Mac: e.g. /dev/ttyUSB0
BAUD = 460800   
# ──────────────────────────────────────────────────────────────────────────────

MAGIC = b'\xFF\xAA\xBB\xCC'


def wait_for_magic(ser):
    """Scan incoming bytes until the 4-byte magic header is found."""
    buf = b''
    while True:
        buf += ser.read(1)
        if buf[-4:] == MAGIC:
            return True
        # Keep buffer small — only need the last 4 bytes
        if len(buf) > 4:
            buf = buf[-4:]


def read_frame(ser):
    """Block until a full JPEG frame arrives, then return raw bytes."""
    if not wait_for_magic(ser):
        return None

    # Next 4 bytes = image length (little-endian uint32)
    raw_len = ser.read(4)
    if len(raw_len) < 4:
        return None
    img_len = struct.unpack('<I', raw_len)[0]

    # Sanity check — QVGA JPEG should never exceed ~50 KB
    if img_len == 0 or img_len > 50_000:
        return None

    jpg = ser.read(img_len)
    if len(jpg) < img_len:
        return None

    return jpg


def main():
    print(f"Opening serial port {PORT} at {BAUD} baud ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=5)
    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Check that the correct PORT is set at the top of this file.")
        return

    print("Connected! Waiting for frames — press Q to quit.")

    while True:
        jpg = read_frame(ser)
        if jpg is None:
            print("Bad/missing frame, retrying ...")
            continue

        # Decode JPEG → OpenCV BGR image
        img_np = np.frombuffer(jpg, dtype=np.uint8)
        frame  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            print("Could not decode JPEG, skipping ...")
            continue

        cv2.imshow("ESP32-cam Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()