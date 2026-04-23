import os
import re

path_md = r'c:\AIP\iad\docs\KHOA_LUAN_TOT_NGHIEP_EN.md'
path_tex = r'c:\AIP\iad\docs\THESIS_FINAL_ACADEMIC.tex'

with open(path_md, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. STRIP FRONT MATTER
start_marker = '# CHAPTER 1: INTRODUCTION'
if start_marker in text:
    main_body = text[text.find(start_marker):]
else:
    main_body = text

# 2. PREAMBLE
preamble = r'''\documentclass[12pt,a4paper,oneside]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[english]{babel}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{setspace}
\usepackage{verbatim}
\usepackage{float}
\usepackage[toc,page]{appendix}

\geometry{a4paper, top=30mm, bottom=25mm, left=35mm, right=20mm}
\onehalfspacing
\captionsetup[table]{position=top, skip=10pt}
\captionsetup[figure]{position=bottom, skip=10pt}

\titleformat{\chapter}[display]{\normalfont\huge\bfseries\centering}{\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titlespacing*{\chapter}{0pt}{0pt}{40pt}

\begin{document}
\begin{titlepage}
    \centering
    \includegraphics[width=0.4\textwidth]{assets/fpt_logo.png}\\[1.5cm]
    {\large \textbf{FPT UNIVERSITY}}\\[2.5cm]
    {\Large \textbf{CAPSTONE PROJECT REPORT}}\\[1cm]
    {\huge \textbf{DEEP LEARNING FOR ANOMALY DETECTION}}\\[0.5cm]
    {\huge \textbf{IN INDUSTRIAL IMAGES}}\\[2cm]
    \begin{minipage}{0.45\textwidth}
        \begin{flushleft} \large \textbf{Research Team:}\\ Nguyen Hoang Minh Son\\ On Nguyen Thien Phuc\\ Nguyen Dang Thai Binh\\ Le Thanh Thao Nhi \end{flushleft}
    \end{minipage}
    \begin{minipage}{0.45\textwidth}
        \begin{flushright} \large \textbf{Supervisors:}\\ Dao Duy Phuong\\ Nguyen Quoc Tien \end{flushright}
    \end{minipage}\\[3cm]
    {\large \textbf{Ho Chi Minh City, 2026}}
\end{titlepage}

\pagenumbering{roman}
\chapter*{Acknowledgements} \addcontentsline{toc}{chapter}{Acknowledgements}
\chapter*{Abstract} \addcontentsline{toc}{chapter}{Abstract}

\tableofcontents
\listoffigures
\listoftables
\clearpage
\pagenumbering{arabic}
'''

# 3. CLEANUP
main_body = re.sub(r'<div.*?>', '', main_body)
main_body = re.sub(r'<\/div>', '', main_body)
main_body = re.sub(r'<br\s*\/?>', '', main_body)

# Fix Special Characters using Unicode codes for safety
main_body = main_body.replace('%', r'\%')
main_body = main_body.replace('≤', r'$\le$').replace('≥', r'$\ge$')
# Replace all variants of mangled/Unicode dashes with standard LaTeX dash
main_body = re.sub(r'[\uFFFD\u2014\u2013\u00AD]', '---', main_body)

# 4. STRUCTURE
main_body = re.sub(r'^# CHAPTER (\d+): (.*?)$', r'\\chapter{\2}', main_body, flags=re.MULTILINE)
main_body = re.sub(r'^## (\d+\.\d+) (.*?)$', r'\\section{\2}', main_body, flags=re.MULTILINE)
main_body = re.sub(r'^### (\d+\.\d+\.\d+) (.*?)$', r'\\subsection{\2}', main_body, flags=re.MULTILINE)

# 5. TABLES
def fix_table(match):
    table_content = match.group(1)
    caption_text = match.group(2)
    lines = [l for l in table_content.strip().split('\n') if l.strip() and not l.strip().startswith('|:')]
    if not lines: return ''
    header = [h.strip() for h in lines[0].strip('|').split('|')]
    tex = f"\\begin{{longtable}}{{{'l'*len(header)}}}\n\\caption{{{caption_text}}} \\\\\n\\toprule\n"
    tex += " & ".join(header) + " \\\\\n\\midrule\n\\endhead\n"
    for row in lines[1:]:
        cells = [c.strip() for c in row.strip('|').split('|')]
        tex += " & ".join(cells) + " \\\\\n"
    tex += "\\bottomrule\n\\end{longtable}\n"
    return tex
main_body = re.sub(r'((?:^\|.*\|(?:\n|$))+)\s*<p align="center"><em>(Table \d+\.\d+:.*?)<\/em><\/p>', fix_table, main_body, flags=re.MULTILINE)

# 6. FIGURES
def fix_figure(match):
    img_path = match.group(2)
    caption_text = match.group(3)
    return f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{{{img_path}}}\n\\caption{{{caption_text}}}\n\\end{{figure}}\n"
main_body = re.sub(r'!\[(.*?)\]\((.*?)\)\s*<p align="center"><em>(Figure \d+\.\d+:.*?)<\/em><\/p>', fix_figure, main_body)

# 7. LISTS
def fix_lists(match):
    items = match.group(0).strip().split('\n')
    tex_list = "\\begin{itemize}\n"
    for item in items:
        item_text = re.sub(r'^[*+-]\s*', '', item.strip())
        tex_list += f"  \\item {item_text}\n"
    tex_list += "\\end{itemize}"
    return tex_list
main_body = re.sub(r'(?:^[*+-]\s+.*(?:\n|$))+', fix_lists, main_body, flags=re.MULTILINE)

# 8. APPENDICES
main_body = main_body.replace('# APPENDICES', '\\begin{appendices}').replace('## Appendix', '\\chapter{Appendix')
main_body += "\n\\end{appendices}"

# 9. FINAL FORMATTING
main_body = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', main_body)
main_body = re.sub(r'_(.*?)_', r'\\textit{\1}', main_body)
main_body = re.sub(r'```\n(.*?)\n```', r'\\begin{verbatim}\n\1\n\\end{verbatim}', main_body, flags=re.DOTALL)

with open(path_tex, 'w', encoding='utf-8') as f:
    f.write(preamble + main_body + "\n\\end{document}")
print("Final Perfect LaTeX Ready.")
