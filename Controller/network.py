import csv
import time
from pathlib import Path
from typing import Dict

from netmiko import ConnectHandler

INVENTORY_FILE   = Path("/Users/mohammedsadathkhan/Downloads/PrivacyGuard/test.csv")
LOG_FILE         = Path("backup_log.txt")


def read_inventory(inventory_path: Path) -> list:
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_path}")
    devices = []
    with inventory_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"ROW {row}")
            devices.append(row)
    return devices

def backup_device(device: Dict[str, str]) -> bool:
    hostname = device.get("hostname") or device.get("ip")
    ip = device.get("ip")
    platform = device.get("platform")
    username = device.get("username")
    password = device.get("password")

    cmd = "copy startup-config ftp://admin:Cisco25@10.141.73.202"

    try:
        conn_params = {
            "device_type": platform,
            "host": ip,
            "username": username,
            "password": password,
            "fast_cli": False,           # safer on some devices
        }
        # For Cisco IOS/IOS-XE/NX-OS: enter enable if needed
        net_conn = ConnectHandler(**conn_params)

        # Some devices need terminal length 0 to avoid paging
        try:
            net_conn.send_command_timing("terminal length 0")
        except Exception:
            pass  # not supported on all platforms

        output = net_conn.send_command(cmd, expect_string=None, use_textfsm=False)
        net_conn.disconnect()
        print(f"Output of the backed up device: {output}")
        return True
    except Exception as e:
        print(f"[{hostname}] SSH issue: {e}")
        return False

# ---------------------------
# Main
# ---------------------------
def main():
    print("=== Config Backup Run Started ===")
    devices = read_inventory(INVENTORY_FILE)
    success = 0
    for d in devices:
        if backup_device(d):
            success += 1
        time.sleep(0.3)  # light pacing to avoid hammering devices

    print(f"=== Completed: {success}/{len(devices)} successful ===")

if __name__ == "__main__":
    main()