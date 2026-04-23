import os
import re
import base64
import markdown
from pathlib import Path

# Paths
DOCS_DIR = Path(r"c:\AIP\iad\docs")
MD_FILE = DOCS_DIR / "KHOA_LUAN_TOT_NGHIEP_EN.md"
HTML_FILE = DOCS_DIR / "KHOA_LUAN_TOT_NGHIEP_EN.html"

# Read markdown
with open(MD_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Clean up the markdown a bit (Enable processing inside divs with markdown="1")
text = text.replace('<div align="center">', '\n\n<div class="centered-content" markdown="1">\n\n')
text = text.replace('</div>', '\n\n</div>\n\n')

# 2. Convert using the most reliable extensions
# arithmatex handles math, extra handles tables/fences, md_in_html allows content inside divs
html_body = markdown.markdown(text, extensions=[
    'extra', 
    'toc', 
    'md_in_html',
    'pymdownx.arithmatex',
    'pymdownx.superfences',
    'pymdownx.betterem'
], extension_configs={
    'pymdownx.arithmatex': {
        'generic': True,
        'smart_dollar': True
    }
})

# 3. Embed images as base64
def embed_img(match):
    tag = match.group(0)
    src_match = re.search(r'src="([^"]+)"', tag)
    if not src_match: return tag
    src = src_match.group(1)
    
    # Resolve path
    p = DOCS_DIR / src
    if not p.exists():
        # Try finding it in assets subfolders
        for sub in ["assets", "assets/results", "assets/demo", "assets/architecture"]:
            tmp = DOCS_DIR / sub / os.path.basename(src)
            if tmp.exists(): p = tmp; break
    
    if p.exists():
        ext = p.suffix.lower()
        mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".svg":"image/svg+xml"}.get(ext,"image/png")
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return tag.replace(src, f"data:{mime};base64,{b64}")
        except: return tag
    return tag

html_body = re.sub(r'<img[^>]+>', embed_img, html_body)

# 4. Final HTML with proper KaTeX and CSS
final_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Thesis Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}},
                {{left: '\\\\[', right: '\\\\]', display: true}},
                {{left: '\\\\(', right: '\\\\)', display: false}}
            ],
            throwOnError: false
        }});"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; color: #333; max-width: 900px; margin: auto; padding: 50px; 
            text-align: justify;
        }}
        h1, h2, h3 {{ color: #1a5f7a; page-break-before: always; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h1:first-child {{ page-break-before: avoid; }}
        img {{ display: block; margin: 20px auto; max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; page-break-inside: avoid; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .centered-content {{ text-align: center; }}
        .arithmatex {{ overflow-x: auto; margin: 1em 0; text-align: center; }}
        @media print {{
            body {{ padding: 0; max-width: 100%; }}
            h1, h2 {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(final_html)
print(f"Success: {HTML_FILE}")
