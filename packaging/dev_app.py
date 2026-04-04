#!/usr/bin/env python3
"""Create a lightweight dev .app bundle for Spotlight.

Instead of the full venvstacks build, this creates a thin .app wrapper
that launches the oMLX menubar app from the project's .venv. Perfect
for development — launches from Spotlight, shows in Dock, etc.

Usage:
    python packaging/dev_app.py          # Creates ~/Applications/oMLX.app
    python packaging/dev_app.py --open   # Creates and opens immediately
"""

import os
import plistlib
import re
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
APP_DIR = Path.home() / "Applications" / "oMLX.app"


def _read_version() -> str:
    version_file = PROJECT_DIR / "omlx" / "_version.py"
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "0.0.0"


def _create_icon(resources_dir: Path):
    """Create app icon from SVG or use a placeholder."""
    icns_path = resources_dir / "AppIcon.icns"

    # Try to render from SVG
    svg_files = list(SCRIPT_DIR.glob("omlx_app/assets/*.svg"))
    if svg_files:
        svg = svg_files[0]
        try:
            # Use sips to convert PNG → iconset → icns
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                iconset = tmpdir / "AppIcon.iconset"
                iconset.mkdir()

                # Try cairosvg first, then rsvg-convert
                png_512 = tmpdir / "icon_512.png"
                try:
                    import cairosvg
                    cairosvg.svg2png(
                        url=str(svg), write_to=str(png_512),
                        output_width=512, output_height=512,
                    )
                except ImportError:
                    subprocess.run(
                        ["rsvg-convert", "-w", "512", "-h", "512", str(svg), "-o", str(png_512)],
                        check=True, capture_output=True,
                    )

                # Create all sizes
                for size in [16, 32, 64, 128, 256, 512]:
                    out = iconset / f"icon_{size}x{size}.png"
                    subprocess.run(
                        ["sips", "-z", str(size), str(size), str(png_512),
                         "--out", str(out)],
                        check=True, capture_output=True,
                    )
                    # @2x
                    if size <= 256:
                        out2x = iconset / f"icon_{size}x{size}@2x.png"
                        s2 = size * 2
                        subprocess.run(
                            ["sips", "-z", str(s2), str(s2), str(png_512),
                             "--out", str(out2x)],
                            check=True, capture_output=True,
                        )

                subprocess.run(
                    ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
                    check=True, capture_output=True,
                )
                print(f"  Icon: created from {svg.name}")
                return
        except Exception as e:
            print(f"  Icon: SVG conversion failed ({e}), using placeholder")

    # Placeholder: create a simple text-based icon
    print("  Icon: no SVG found, skipping (app will use default icon)")


def create_app(open_after: bool = False):
    """Create the dev .app bundle."""
    version = _read_version()
    print(f"Creating oMLX.app (dev) v{version}")

    if not VENV_PYTHON.exists():
        print(f"ERROR: .venv not found at {VENV_PYTHON}")
        print("Run: python -m venv .venv && .venv/bin/pip install -e .")
        sys.exit(1)

    # Clean existing
    if APP_DIR.exists():
        import shutil
        shutil.rmtree(APP_DIR)

    # Create structure
    contents = APP_DIR / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    # 1. Info.plist
    plist = {
        "CFBundleName": "oMLX",
        "CFBundleDisplayName": "oMLX",
        "CFBundleIdentifier": "com.omlx.app.dev",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "CFBundleExecutable": "oMLX",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "LSMinimumSystemVersion": "15.0",
        "LSUIElement": True,  # Menubar app — no Dock icon
        "NSHighResolutionCapable": True,
        "LSArchitecturePriority": ["arm64"],
    }

    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist, f)

    # 2. Launcher script
    launcher = macos / "oMLX"
    launcher.write_text(f"""#!/bin/bash
# oMLX dev launcher — uses project .venv
export PYTHONPATH="{PROJECT_DIR}"
exec "{VENV_PYTHON}" -m omlx_app "$@"
""")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)

    # 3. Icon
    _create_icon(resources)

    # 4. Ad-hoc code sign (required for Gatekeeper on macOS 15+)
    try:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(APP_DIR)],
            check=True, capture_output=True,
        )
        print("  Codesign: ad-hoc signed")
    except Exception:
        print("  Codesign: skipped (may not launch from Spotlight)")

    # 5. Register with Launch Services (makes it findable by Spotlight)
    subprocess.run(
        ["/System/Library/Frameworks/CoreServices.framework/Frameworks/"
         "LaunchServices.framework/Support/lsregister",
         "-f", str(APP_DIR)],
        capture_output=True,
    )

    print(f"\noMLX.app created at: {APP_DIR}")
    print("Spotlight should find it within a few seconds.")
    print("Search for 'oMLX' in Spotlight (Cmd+Space).")

    if open_after:
        subprocess.run(["open", str(APP_DIR)])
        print("App launched!")


if __name__ == "__main__":
    open_after = "--open" in sys.argv
    create_app(open_after=open_after)
