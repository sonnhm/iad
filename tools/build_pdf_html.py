import json
import os


def build_html(md_path, html_path):
    if not os.path.exists(md_path):
        print(f"File not found: {md_path}")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Dùng json.dumps để encode trọn vẹn dấu nháy, xuống dòng, ký tự đặc biệt
    encoded_md = json.dumps(md_text)

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Thesis Report - Industrial Anomaly Detection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    <style>
        body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
            font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
            background-color: white;
        }}
        @media print {{
            body {{
                padding: 0;
            }}
            .markdown-body {{
                font-size: 13pt;
            }}
            /* Canh lề biểu đồ để không bị tràn giấy A4 */
            .mermaid svg {{
               max-width: 100% !important;
               height: auto;
            }}
            /* Tránh ngắt dòng bảng biểu */
            table {{ page-break-inside:auto }}
            tr    {{ page-break-inside:avoid; page-break-after:auto }}
            thead {{ display:table-header-group }}
            tfoot {{ display:table-footer-group }}
        }}
        .markdown-body img {{
             max-width: 100%;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head>
<body class="markdown-body">
    <div id="content"></div>
    <script>
        const rawMd = {encoded_md};
        
        // Cấu hình Marked.js để bắt được thẻ Mermaid
        const renderer = new marked.Renderer();
        const defaultCodeRenderer = renderer.code;
        renderer.code = function(code, language, isEscaped) {{
            if (language === 'mermaid') {{
                return '<div class="mermaid">' + code + '</div>';
            }}
            return defaultCodeRenderer.call(this, code, language, isEscaped);
        }};
        marked.setOptions({{ renderer: renderer }});
        
        // Render Markdown
        document.getElementById('content').innerHTML = marked.parse(rawMd);
        
        // Gọi Mermaid vẽ Sơ đồ khối (Flowchart)
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"  [SUCCESS] Đã tạo file: {html_path}")


print("==========================================================")
print("     CÔNG CỤ KẾT XUẤT KHÓA LUẬN (RENDER THESIS PDF)       ")
print("==========================================================")
build_html("docs/KHOA_LUAN_TOT_NGHIEP.md", "docs/THESIS_VN.html")
build_html("docs/KHOA_LUAN_TOT_NGHIEP_EN.md", "docs/THESIS_EN.html")
print("----------------------------------------------------------")
print("HƯỚNG DẪN IN PDF:")
print("1. Nhấp đúp mở file THESIS_VN.html bằng Google Chrome/Edge.")
print("2. Chờ 2 giây để Sơ đồ khối (Mermaid) vẽ xong.")
print("3. Bấm tổ hợp phím [Ctrl + P].")
print("4. Chỉnh cấu hình: Lề (Margins) -> Mặc định (Default).")
print("5. Nhấn Lưu thành PDF (Save as PDF) và gửi cho Giáo viên!")
print("==========================================================")
