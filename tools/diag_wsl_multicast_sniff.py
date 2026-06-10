"""Phase-2-Diagnose: Multicast-Sniffer in WSL.

Lauscht 15 Sekunden auf FastDDS' default Discovery-Multicast-Group
(239.255.0.1:7400 für Domain 0). Wenn Pakete reinkommen, weiss man:
  - Sim publisht aktuell DDS-Discovery-Pakete
  - Mirrored networking traegt Multicast aus Windows in WSL hinein
  - Discovery-Schicht selbst funktioniert auf TCP/UDP-Level
Wenn keine Pakete reinkommen:
  - Entweder Sim sendet nicht (Sim restart) oder
  - mirrored networking blockt cross-host multicast (WSL Issue #12122)

Aufruf in WSL:
    python3 /mnt/c/dev/tetrabot_sim/tools/diag_wsl_multicast_sniff.py

Kein sudo noetig (Multicast-IGMP-Subscription darf jeder).
"""
from __future__ import annotations

import socket
import struct
import sys
import time

GROUP = "239.255.0.1"   # FastDDS default Discovery multicast group
PORT = 7400             # Domain 0 builtin port (PB + DG*0 + d0)
LISTEN_SECONDS = 15


def main() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    sock.bind(("", PORT))

    mreq = struct.pack("=4sl", socket.inet_aton(GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(1.0)

    print(f"Lausche {LISTEN_SECONDS}s auf Multicast {GROUP}:{PORT} ...")
    sys.stdout.flush()

    deadline = time.time() + LISTEN_SECONDS
    packets = 0
    senders: dict[str, int] = {}
    rtps_packets = 0

    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        packets += 1
        senders[addr[0]] = senders.get(addr[0], 0) + 1
        # FastDDS-Pakete starten mit "RTPS" magic bytes
        if data[:4] == b"RTPS":
            rtps_packets += 1

    print()
    print(f"=== Ergebnis nach {LISTEN_SECONDS}s ===")
    print(f"Total Pakete:  {packets}")
    print(f"RTPS-Pakete:   {rtps_packets}  (Magic-Bytes 'RTPS' = FastDDS/CycloneDDS)")
    print(f"Sender-IPs:")
    for ip, n in sorted(senders.items(), key=lambda kv: -kv[1]):
        print(f"  {ip}  -> {n} Pakete")

    print()
    if rtps_packets > 0:
        print("OK Multicast-Cross-Host funktioniert — DDS-Discovery erreicht WSL.")
        print("   -> Wenn rclpy/Bridge trotzdem keine Topics sieht, ist's eine")
        print("      hoehere Schicht (FastDDS init in WSL, Domain-Mismatch, RMW).")
        return 0
    elif packets > 0:
        print("PARTIAL: Multicast-Pakete kommen an, aber keine RTPS — andere App?")
        return 2
    else:
        print("FAIL Keine Multicast-Pakete in 15s.")
        print("   -> Entweder Sim sendet nicht, oder mirrored-networking blockt")
        print("      Multicast aus Windows nach WSL (WSL Issue #12122).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
