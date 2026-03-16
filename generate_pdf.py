#!/usr/bin/env python3
"""Convert ARCHITECTURE_DNA.md to PDF."""
import markdown
from weasyprint import HTML

with open("ARCHITECTURE_DNA.md", "r") as f:
    md_content = f.read()

html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code", "codehilite"])

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 2cm;
}}
body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #1a1a1a;
}}
h1 {{
    font-size: 22pt;
    color: #0d1117;
    border-bottom: 3px solid #0969da;
    padding-bottom: 8px;
    margin-top: 24px;
}}
h2 {{
    font-size: 16pt;
    color: #0d1117;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 4px;
    margin-top: 20px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 13pt;
    color: #24292f;
    margin-top: 16px;
    page-break-after: avoid;
}}
h4 {{
    font-size: 11pt;
    color: #24292f;
    margin-top: 12px;
    page-break-after: avoid;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}
th {{
    background-color: #0969da;
    color: white;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
}}
td {{
    padding: 5px 10px;
    border: 1px solid #d0d7de;
}}
tr:nth-child(even) {{
    background-color: #f6f8fa;
}}
code {{
    background-color: #f6f8fa;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 9pt;
}}
pre {{
    background-color: #161b22;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}}
pre code {{
    background: none;
    color: #e6edf3;
    padding: 0;
}}
hr {{
    border: none;
    border-top: 2px solid #d0d7de;
    margin: 20px 0;
}}
blockquote {{
    border-left: 4px solid #0969da;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #f6f8fa;
}}
strong {{
    color: #0d1117;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

output_path = "ARCHITECTURE_DNA.pdf"
HTML(string=full_html).write_pdf(output_path)
print(f"PDF generated: {output_path}")
