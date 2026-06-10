"""Sanity-checks on the TETRABot URDF without external dependencies.

Usage:
    python tools/validate_urdf.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_PATH = Path(__file__).resolve().parents[1] / "urdf" / "tetrabot.urdf"


def main() -> int:
    if not URDF_PATH.exists():
        print(f"FAIL: URDF not found at {URDF_PATH}", file=sys.stderr)
        return 1

    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    links = root.findall("link")
    joints = root.findall("joint")
    print(f"Robot: {root.attrib.get('name')}")
    print(f"  Links:  {len(links)}")
    print(f"  Joints: {len(joints)}")

    issues: list[str] = []
    total_mass = 0.0

    for link in links:
        name = link.attrib["name"]
        inertial = link.find("inertial")
        if inertial is None:
            # zero-mass tf anchors are intentional (camera frames)
            continue
        mass_el = inertial.find("mass")
        if mass_el is None or "value" not in mass_el.attrib:
            issues.append(f"link '{name}' has <inertial> but no mass value")
            continue
        m = float(mass_el.attrib["value"])
        total_mass += m

    print(f"  Total mass (sum of <inertial><mass>): {total_mass:.3f} kg")

    parent_count: dict[str, int] = {}
    for joint in joints:
        jname = joint.attrib["name"]
        jtype = joint.attrib["type"]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            issues.append(f"joint '{jname}' missing <parent> or <child>")
            continue
        cl = child.attrib["link"]
        parent_count[cl] = parent_count.get(cl, 0) + 1
        if jtype in ("prismatic", "revolute"):
            limit = joint.find("limit")
            if limit is None:
                issues.append(f"joint '{jname}' is {jtype} but has no <limit>")

    multi_parent = [c for c, n in parent_count.items() if n > 1]
    if multi_parent:
        issues.append(f"links with >1 parent joint: {multi_parent}")

    link_names = {l.attrib["name"] for l in links}
    children = {j.find("child").attrib["link"] for j in joints if j.find("child") is not None}
    roots = link_names - children
    print(f"  Root link(s): {sorted(roots)}")
    if len(roots) != 1:
        issues.append(f"expected exactly 1 root link, found {len(roots)}: {sorted(roots)}")

    actuated = [j.attrib["name"] for j in joints if j.attrib["type"] in ("prismatic", "revolute")]
    passive = [j.attrib["name"] for j in joints if j.attrib["type"] == "continuous"]
    fixed = [j.attrib["name"] for j in joints if j.attrib["type"] == "fixed"]
    print(f"  Actuated joints (prismatic/revolute): {len(actuated)} -> {actuated}")
    print(f"  Passive continuous joints:            {len(passive)}")
    print(f"  Fixed joints:                         {len(fixed)}")

    if issues:
        print("\nISSUES:")
        for i in issues:
            print(f"  - {i}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
