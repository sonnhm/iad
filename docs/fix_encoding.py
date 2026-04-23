import os

path = r'c:\AIP\iad\docs\KHOA_LUAN_TOT_NGHIEP_EN.html'

# Read as bytes to detect and fix mojibake
with open(path, 'rb') as f:
    content = f.read()

# The mojibake "â”" typically happens when UTF-8 bytes are interpreted as CP1252
# We need to ensure the file is clean UTF-8. 
# If it was saved by PowerShell with a BOM, we strip it.
if content.startswith(b'\xef\xbb\xbf'):
    content = content[3:]

try:
    # Try decoding as UTF-8. If it works, the file is already UTF-8.
    # The mojibake in the browser might be because of the BOM or missing meta tag.
    text = content.decode('utf-8')
except UnicodeDecodeError:
    # If UTF-8 fails, it was definitely mangled. Decode as CP1252 and re-encode.
    text = content.decode('cp1252')

# Ensure the HTML has a UTF-8 meta tag if missing
if '<meta charset="utf-8">' not in text.lower():
    text = text.replace('<head>', '<head>\n    <meta charset="utf-8">')

# Re-write clean UTF-8 without BOM
with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(text)

print(f"Fixed: {path} is now clean UTF-8.")
