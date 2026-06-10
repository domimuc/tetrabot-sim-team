"""Phase-1-Diagnose: Windows-side rclpy listener.

Nutzt Isaac Sim's bundled ROS2-Humble (selbe DDS-Lib wie der Sim selbst,
KEINE Cross-Host-Schicht dazwischen) um zu probieren ob die /tetrabot/*-
Topics auf Windows-Seite gerade DDS-sichtbar sind.

Aufruf via Isaac Sim's python.bat:
    "C:\\isaac-sim\\python.bat" tools\\diag_windows_listener.py

Ergebnis-Interpretation:
  - Listet /tetrabot/camera/{rgb,depth,camera_info}
    -> Sim publisht aktuell, DDS auf Windows-Seite OK
       -> Problem ist Cross-Host (WSL kann's nicht aufpicken)
  - Listet nur /parameter_events, /rosout
    -> Sim publisht ZWAR die Bridge-Setup, aber kein Render
       -> ROS-Graph in Sim aktiv aber Frames erreichen die Helper nicht
  - Listet gar nichts
    -> Sim's bundled rclpy hat Discovery-Probleme schon lokal
       -> Sim restart noetig
"""
from __future__ import annotations

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    print("FEHLER: rclpy nicht importierbar. Wird via launch.bat-Env aufgerufen?")
    print(f"sys.path: {sys.path[:5]}")
    sys.exit(1)


def main() -> int:
    rclpy.init()
    node = Node("tetrabot_diag_listener")
    print("rclpy initialisiert. Warte 8 s auf Discovery...")
    sys.stdout.flush()

    # FastDDS braucht Multicast-Heartbeats; default-Period ist ~5 s.
    # 8 s gibt zwei Discovery-Cycles Zeit.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    topics = node.get_topic_names_and_types()
    nodes = node.get_node_names()

    print(f"\n=== {len(nodes)} Knoten sichtbar ===")
    for name in sorted(nodes):
        print(f"  {name}")

    print(f"\n=== {len(topics)} Topics sichtbar ===")
    for name, types in sorted(topics):
        marker = "  >>> " if name.startswith("/tetrabot/") else "      "
        print(f"{marker}{name}  {types}")

    tetrabot_topics = [n for n, _ in topics if n.startswith("/tetrabot/")]
    print()
    if tetrabot_topics:
        print(f"OK Sim publisht auf DDS Windows-Seite: {len(tetrabot_topics)} Topics")
        rc = 0
    else:
        print("FAIL Sim NICHT sichtbar von Windows-side rclpy aus.")
        print("     -> Sim DDS-Layer hat Probleme; Sim restart noetig.")
        rc = 1

    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
