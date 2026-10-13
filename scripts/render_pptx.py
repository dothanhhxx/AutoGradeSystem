"""Export every slide of a PPTX to PNG using PowerPoint COM (Windows only).

Usage: python scripts/render_pptx.py deck.pptx out_dir [width_px]
"""
import os
import sys

import win32com.client  # type: ignore  (pip install pywin32)


def main() -> None:
    src = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    os.makedirs(out, exist_ok=True)

    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(src, WithWindow=False)
    try:
        ratio = pres.PageSetup.SlideHeight / pres.PageSetup.SlideWidth
        for i, _ in enumerate(pres.Slides, start=1):
            pres.Slides(i).Export(os.path.join(out, f"slide_{i:02d}.png"), "PNG", width, int(width * ratio))
    finally:
        pres.Close()
        app.Quit()
    print(f"Rendered to {out}")


if __name__ == "__main__":
    main()
