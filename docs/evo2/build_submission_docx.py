"""Render the canonical Evo² Markdown report as a polished, image-embedded DOCX."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
from bs4 import BeautifulSoup


TITLE = "Evo²-Ecosystem: Sakana AI Project, Mark Moussa"

CSS = """
@page { size: A4; margin: 0.72in; }
body {
  max-width: 7.05in; margin: 0 auto; color: #111827;
  font-family: Arial, Helvetica, sans-serif; font-size: 10.4pt;
  line-height: 1.37;
}
h1 { color: #0f172a; font-size: 25pt; margin: 0 0 12pt; }
h2 {
  color: #0f3f75; font-size: 17pt; margin: 22pt 0 8pt;
  border-bottom: 1.2pt solid #cbd5e1; padding-bottom: 3pt;
}
h3 {
  color: #1e3a5f; font-size: 13.5pt; margin: 16pt 0 6pt;
  page-break-after: avoid;
}
h4 { color: #334155; font-size: 11.5pt; margin: 13pt 0 5pt; }
p { margin: 5.5pt 0; }
blockquote {
  margin: 10pt 16pt; padding: 8pt 12pt;
  border-left: 4pt solid #2563eb; background: #eff6ff;
}
blockquote p { margin: 0; font-weight: bold; }
table {
  width: 100%; table-layout: fixed; border-collapse: collapse;
  margin: 9pt 0 13pt; font-size: 8.5pt; page-break-inside: auto;
}
th { background: #e8eef6; color: #0f172a; font-weight: bold; }
th, td {
  border: 0.6pt solid #94a3b8; padding: 4pt 5pt;
  vertical-align: top; overflow-wrap: anywhere; word-wrap: break-word;
}
tr { page-break-inside: avoid; }
pre {
  background: #f1f5f9; border: 0.6pt solid #cbd5e1; padding: 7pt;
  font-size: 7.5pt; line-height: 1.25; white-space: pre-wrap;
  page-break-inside: avoid;
}
code { font-family: "Liberation Mono", Consolas, monospace; font-size: 0.91em; }
p code { background: #f1f5f9; padding: 1pt 2pt; }
img {
  display: block; max-width: 100%; max-height: 7.2in; height: auto;
  margin: 11pt auto 4pt; page-break-inside: avoid;
}
.figure-image { page-break-after: avoid; }
.figure-caption { page-break-before: avoid; page-break-inside: avoid; }
p > em {
  display: block; color: #475569; font-size: 8.8pt;
  line-height: 1.25; margin: 0 10pt 10pt;
}
li { margin: 2.5pt 0; }
ul, ol { margin-top: 4pt; margin-bottom: 7pt; }
"""


def _inline_images(html: str, source_dir: Path) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for image in soup.find_all("img"):
        source = image.get("src")
        if not source or source.startswith(("data:", "http://", "https://")):
            continue
        path = (source_dir / source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing report image: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        image["src"] = f"data:{mime};base64,{encoded}"
        image["width"] = "650"
        if image.parent is not None and image.parent.name == "p":
            image.parent["class"] = ["figure-image"]
    for paragraph in soup.find_all("p"):
        if paragraph.find("em", recursive=False) is not None:
            paragraph["class"] = ["figure-caption"]
    return str(soup)


def _render_html(source: Path) -> str:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=("tables", "fenced_code", "sane_lists"),
        output_format="html5",
    )
    body = _inline_images(body, source.parent)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>{body}</body>
</html>
"""


def _finalize_document(converted: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(converted, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    output = arguments.output.resolve()
    libreoffice = shutil.which("libreoffice")
    if libreoffice is None:
        raise RuntimeError("LibreOffice is required to render the final DOCX")

    with tempfile.TemporaryDirectory(prefix="evo2-docx-") as temporary:
        root = Path(temporary)
        html_path = root / "sakana-submission.html"
        converted_dir = root / "converted"
        profile = root / "libreoffice-profile"
        converted_dir.mkdir()
        profile.mkdir()
        html_path.write_text(_render_html(source), encoding="utf-8")

        subprocess.run(
            (
                libreoffice,
                f"-env:UserInstallation={profile.resolve().as_uri()}",
                "--headless",
                "--convert-to",
                "docx:Office Open XML Text",
                "--outdir",
                str(converted_dir),
                str(html_path),
            ),
            check=True,
        )
        converted = converted_dir / "sakana-submission.docx"
        if not converted.is_file():
            raise RuntimeError("LibreOffice did not produce the expected DOCX")
        _finalize_document(converted, output)


if __name__ == "__main__":
    main()
