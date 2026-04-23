#!/usr/bin/env python3
"""
Convert KHOA_LUAN_TOT_NGHIEP_EN.md to a complete, publication-ready LaTeX document.
Strategy:
  1. Use pandoc for core MD→LaTeX body conversion
  2. Post-process the body to fix academic formatting
  3. Wrap with a professional preamble for Overleaf
"""
import subprocess
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MD = os.path.join(SCRIPT_DIR, "KHOA_LUAN_TOT_NGHIEP_EN.md")
OUTPUT_TEX = os.path.join(SCRIPT_DIR, "THESIS_COMPLETE.tex")

# ── Step 1: Pandoc conversion ──────────────────────────────────────
def pandoc_convert(md_path: str) -> str:
    """Convert markdown to LaTeX body using pandoc."""
    cmd = [
        "pandoc", md_path,
        "-f", "markdown+tex_math_dollars+pipe_tables+raw_html",
        "-t", "latex",
        "--wrap=none",
        "--columns=9999",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"Pandoc stderr: {result.stderr}")
    return result.stdout

# ── Step 2: Post-processing ────────────────────────────────────────
def postprocess(body: str) -> str:
    """Fix and enhance the pandoc LaTeX output for academic quality."""

    # 2a. Fix chapter/section hierarchy
    # Pandoc sometimes converts # to \section or just \textbf within a list
    # First, handle the actual headers
    body = re.sub(r'\\section\{CHAPTER (\d+):', r'\\chapter{CHAPTER \1:', body)
    body = re.sub(r'\\section\{(REFERENCES)\}', r'\\chapter*{\1}\n\\addcontentsline{toc}{chapter}{\1}', body)
    body = re.sub(r'\\section\{(APPENDICES)\}', r'\\chapter*{\1}\n\\addcontentsline{toc}{chapter}{\1}', body)
    body = re.sub(r'\\section\{(LIST OF FIGURES)\}', r'\\chapter*{\1}\n\\addcontentsline{toc}{chapter}{\1}', body)
    body = re.sub(r'\\section\{(LIST OF TABLES)\}', r'\\chapter*{\1}\n\\addcontentsline{toc}{chapter}{\1}', body)
    body = re.sub(r'\\section\{(DEFINITIONS AND ACRONYMS)\}', r'\\chapter*{\1}\n\\addcontentsline{toc}{chapter}{\1}', body)

    # If pandoc didn't use \section but instead left them as items or bold text (common in some MD flavors)
    body = re.sub(r'\\textbf\{CHAPTER (\d+): ([^}]+)\}', r'\\chapter{CHAPTER \1: \2}', body)

    # Fix Appendix sections
    body = re.sub(r'\\subsection\{(Appendix [A-Z]:)', r'\\section{\\1', body)

    # 2b. Fix figure environments from pandoc's raw HTML conversion
    # Replace pandoc's inline image references with proper figure environments
    def fix_figure(match):
        full = match.group(0)
        # Extract path and caption
        path_match = re.search(r'\\includegraphics\{([^}]+)\}', full)
        cap_match = re.search(r'Figure \d+\.\d+:([^}]+)', full)
        if not cap_match:
            cap_match = re.search(r'\\caption\{([^}]+)\}', full)

        path = path_match.group(1) if path_match else "image"
        caption = cap_match.group(0).strip() if cap_match else ""

        # Clean caption
        caption = caption.replace('\\\\', ' ').strip()
        if caption.endswith('.'):
            pass
        elif caption:
            caption += '.'

        return (
            f"\\begin{{figure}}[htbp]\n"
            f"\\centering\n"
            f"\\includegraphics[width=0.85\\textwidth]{{{path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\end{{figure}}"
        )

    body = re.sub(
        r'\\begin\{figure\}.*?\\end\{figure\}',
        fix_figure,
        body,
        flags=re.DOTALL
    )

    # 2c. Fix table captions - move <p> style captions into proper \caption
    # Pattern: table followed by standalone caption text
    def fix_table_caption(match):
        table_content = match.group(1)
        caption_text = match.group(2).strip()
        # Remove "Table X.Y: " prefix for label, keep for caption
        label_match = re.search(r'Table ([A-Z]?\d+\.\d+[a-e]?)', caption_text)
        label = ""
        if label_match:
            label_id = label_match.group(1).replace('.', '_').lower()
            label = f"\\label{{tab:{label_id}}}"

        return (
            f"\\begin{{table}}[htbp]\n"
            f"\\centering\n"
            f"\\caption{{{caption_text}}}\n"
            f"{label}\n"
            f"{table_content}\n"
            f"\\end{{table}}"
        )

    # Fix standalone table captions that pandoc leaves as plain text
    body = re.sub(
        r'(\\begin\{longtable\}.*?\\end\{longtable\})\s*\n\s*\n\s*(Table [A-Z]?\d+\.\d+[a-e]?:[^\n]+)',
        fix_table_caption,
        body,
        flags=re.DOTALL
    )

    # 2d. Clean up HTML artifacts that pandoc may leave
    body = re.sub(r'<div[^>]*>', '', body)
    body = re.sub(r'</div>', '', body)
    body = re.sub(r'<br\s*/?>', r'\\\\', body)
    body = re.sub(r'<em>(.*?)</em>', r'\\emph{\1}', body)
    body = re.sub(r'<p[^>]*>(.*?)</p>', r'\1', body, flags=re.DOTALL)
    body = re.sub(r'<img[^>]*>', '', body)

    # 2e. Fix horizontal rules → clean separation
    body = re.sub(
        r'\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}',
        r'\\bigskip\\noindent\\hrulefill\\bigskip',
        body
    )

    # 2f. Ensure code blocks use lstlisting
    body = re.sub(r'\\begin\{verbatim\}', r'\\begin{lstlisting}', body)
    body = re.sub(r'\\end\{verbatim\}', r'\\end{lstlisting}', body)

    # Fix Shaded environments from pandoc
    body = re.sub(r'\\begin\{Shaded\}\s*\\begin\{Highlighting\}\[\]',
                  r'\\begin{lstlisting}', body)
    body = re.sub(r'\\end\{Highlighting\}\s*\\end\{Shaded\}',
                  r'\\end{lstlisting}', body)

    # Clean highlighting commands inside lstlisting
    for cmd in ['KeywordTok', 'NormalTok', 'BuiltInTok', 'SelfTok',
                'OperatorTok', 'CommentTok', 'StringTok', 'ImportTok',
                'ControlFlowTok', 'VariableTok', 'OtherTok', 'FunctionTok',
                'AttributeTok', 'DecValTok', 'SpecialCharTok']:
        body = re.sub(rf'\\{cmd}\{{([^}}]*)\}}', r'\1', body)

    # 2g. Fix \tightlist
    body = body.replace('\\tightlist\n', '')
    body = body.replace('\\tightlist', '')

    # 2h. Remove \def\labelenumi patterns
    body = re.sub(r'\\def\\labelenumi\{[^}]*\}\s*\n?', '', body)

    # 2i. Fix the cover page - make it a proper title page
    # Remove pandoc's conversion of the cover and replace with proper LaTeX title page
    # This is handled in the preamble via \maketitle

    # 2j. Fix ≤ and ≥ symbols
    body = body.replace('≤', '$\\leq$')
    body = body.replace('≥', '$\\geq$')
    body = body.replace('×', '$\\times$')
    body = body.replace('→', '$\\rightarrow$')
    body = body.replace('←', '$\\leftarrow$')

    # 2k. Fix checkmarks and crosses in tables
    body = body.replace('✅', '\\checkmark')
    body = body.replace('❌', '\\texttimes')
    body = body.replace('⚠️', '\\textbf{!}')

    # 2l. Fix \textasciitilde to ~
    body = body.replace('\\textasciitilde{}', '\\texttildelow{}')
    body = re.sub(r'\\textasciitilde(\d)', r'${\\sim}$\1', body)

    # 2m. Fix nested quote environments
    body = re.sub(r'\\begin\{quote\}\s*\\textbf\{', r'\\begin{quote}\n\\textbf{', body)

    return body

# ── Step 3: Preamble ───────────────────────────────────────────────
PREAMBLE = r"""\documentclass[12pt, a4paper, oneside]{report}

% ── Encoding & Language ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}

% ── Page Geometry ──
\usepackage[
  top=2.5cm,
  bottom=2.5cm,
  left=3.0cm,
  right=2.5cm
]{geometry}

% ── Typography ──
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{setspace}
\onehalfspacing

% ── Math ──
\usepackage{amsmath, amssymb, amsfonts, mathtools}
\usepackage{bm}

% ── Graphics & Floats ──
\usepackage{graphicx}
\graphicspath{{./assets/}{./}}
\usepackage{float}
\usepackage[font=small, labelfont=bf, skip=8pt]{caption}
\usepackage{subcaption}

% ── Tables ──
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{tabularx}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

% ── Code Listings ──
\usepackage{listings}
\usepackage{xcolor}
\definecolor{codebg}{HTML}{F5F5F5}
\definecolor{codeframe}{HTML}{CCCCCC}
\definecolor{codegreen}{HTML}{2E7D32}
\definecolor{codeblue}{HTML}{1565C0}
\definecolor{codegray}{HTML}{757575}
\lstset{
  backgroundcolor=\color{codebg},
  frame=single,
  rulecolor=\color{codeframe},
  basicstyle=\ttfamily\small,
  keywordstyle=\color{codeblue}\bfseries,
  commentstyle=\color{codegreen}\itshape,
  stringstyle=\color{codegray},
  breaklines=true,
  breakatwhitespace=false,
  tabsize=4,
  showstringspaces=false,
  numbers=none,
  xleftmargin=1em,
  xrightmargin=1em,
  aboveskip=1em,
  belowskip=1em,
}

% ── Links & References ──
\usepackage[
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black
]{hyperref}
\usepackage{bookmark}

% ── Headers & Footers ──
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% ── Misc ──
\usepackage{enumitem}
\setlist{nosep, leftmargin=2em}
\usepackage{pifont}
\newcommand{\checkmark}{\ding{51}}
\newcommand{\texttimes}{\ding{55}}
\usepackage{textcomp}

% ── Prevent overfull hbox ──
\tolerance=1000
\emergencystretch=3em
\hfuzz=2pt

% ── Fix for pandoc ──
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\texttildelow}{{\raise-.5ex\hbox{\~{}}}}

% ══════════════════════════════════════════════════════════════════
\begin{document}

% ── Title Page ──────────────────────────────────────────────────
\begin{titlepage}
\begin{center}
\vspace*{1cm}

\includegraphics[width=0.25\textwidth]{assets/fpt_logo.png}

\vspace{1.5cm}

{\Large\textbf{FPT UNIVERSITY}}\\[0.3cm]
{\large Faculty of Information Technology}

\vspace{2cm}

{\LARGE\textbf{CAPSTONE PROJECT REPORT}}

\vspace{1cm}

{\Huge\textbf{DEEP LEARNING FOR\\ANOMALY DETECTION IN\\INDUSTRIAL IMAGES}}

\vspace{2cm}

\textbf{Project Code:} SP26AI55 \hspace{2cm} \textbf{Group Code:} GSP26AI27

\vspace{1cm}

\textbf{Research Team:}\\[0.3cm]
\begin{tabular}{ll}
\toprule
\textbf{Full Name} & \textbf{Student ID} \\
\midrule
NGUYEN HOANG MINH SON & SE151025 \\
ON NGUYEN THIEN PHUC & SE172629 \\
NGUYEN DANG THAI BINH & SE183718 \\
LE THANH THAO NHI & SE172759 \\
\bottomrule
\end{tabular}

\vspace{1.5cm}

\textbf{Supervisor:} DAO DUY PHUONG \& NGUYEN QUOC TIEN

\vfill

\emph{Ho Chi Minh City, 2026}

\end{center}
\end{titlepage}

% ── Acknowledgements ────────────────────────────────────────────
\chapter*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}

We would like to express our sincere gratitude to our academic advisor Mr.~DAO DUY PHUONG and Mr.~NGUYEN QUOC TIEN for their dedicated guidance and for providing a solid foundation of knowledge throughout the execution of this Graduation Project. Their expertise, constructive feedback, and dedication have been essential in shaping the direction and quality of our work. We are sincerely thankful for the time and effort they devoted to mentoring us, helping us overcome challenges, and motivating us to push beyond our limits. Without their continued assistance, this project would not have been possible. We would like to thank our friends and families for their patience, understanding, and encouragement throughout this journey. Finally, we would like to thank our university providing the academic environment, resources, and opportunities necessary for us to complete this research.

% ── Abstract ────────────────────────────────────────────────────
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

Artificial Intelligence optical inspection is rapidly becoming a mandatory standard in industrial manufacturing, completely replacing the inconsistent and error-prone manual measurement methods. This thesis proposes and develops a comparative multi-model, Unsupervised Anomaly Detection (UAD) system executed concurrently that operates entirely on Nominal Manifolds. The system deploys a comparative triad of models: fundamental machine learning (OC-SVM), structural approximations via generative autoencoders, and its apex architecture, the memory-based Enhanced PatchCore. By thoroughly exploiting a \texttt{CustomResNet18} backbone fortified with Knowledge Distillation, along with a deterministic \texttt{k-Center Greedy Coreset} subsampling strategy combined with Locality-Sensitive Hashing (LSH), the system unlocks unprecedented real-time processing capabilities. This research rigorously evaluates latency, VRAM optimization, and quantitative efficacy through an exhaustive Ablation Study on the MVTec AD dataset (15 categories). Furthermore, we establish mathematical boundaries for error rates and numerical stability. The finalized results firmly establish the industrial superiority and on-premise viability of our optimized framework for next-generation smart manufacturing lines.

% ── Table of Contents ───────────────────────────────────────────
\tableofcontents
\listoffigures
\addcontentsline{toc}{chapter}{List of Figures}
\listoftables
\addcontentsline{toc}{chapter}{List of Tables}

"""

POSTAMBLE = r"""
\end{document}
"""

# ── Main ───────────────────────────────────────────────────────────
def main():
    print("[1/3] Running pandoc conversion...")
    body = pandoc_convert(INPUT_MD)

    print("[2/3] Post-processing LaTeX body...")
    body = postprocess(body)

    # Remove the cover page, acknowledgements, abstract, and TOC sections
    # that pandoc generated (we handle these in the preamble)
    # Find where Chapter 1 starts
    ch1_match = re.search(r'\\(?:chapter|section|textbf)\{CHAPTER 1:', body)
    if ch1_match:
        # Find content before Chapter 1 and remove pandoc's cover/abstract/TOC
        pre_ch1 = body[:ch1_match.start()]
        post_ch1 = body[ch1_match.start():]

        # Keep only the DEFINITIONS AND ACRONYMS table from pre-chapter content
        defs_match = re.search(
            r'(\\chapter\*\{DEFINITIONS AND ACRONYMS\}.*?)(?=\\chapter|$)',
            pre_ch1,
            flags=re.DOTALL
        )
        if not defs_match:
            # Try subsection variant
            defs_match = re.search(
                r'(\\subsection\{DEFINITIONS AND ACRONYMS\}.*?)(?=\\section|\\chapter|\\subsection\{)',
                pre_ch1,
                flags=re.DOTALL
            )

        defs_section = ""
        if defs_match:
            defs_section = defs_match.group(1)
            # Convert to chapter* if it's a subsection
            defs_section = defs_section.replace(
                '\\subsection{DEFINITIONS AND ACRONYMS}',
                '\\chapter*{DEFINITIONS AND ACRONYMS}\n\\addcontentsline{toc}{chapter}{Definitions and Acronyms}'
            )

        # Also extract List of Figures/Tables entries if present as proper lists
        body = defs_section + "\n\n" + post_ch1
    else:
        print("WARNING: Could not find Chapter 1 marker in pandoc output!")

    print("[3/3] Writing complete LaTeX document...")
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(PREAMBLE)
        f.write(body)
        f.write(POSTAMBLE)

    # Verify
    line_count = sum(1 for _ in open(OUTPUT_TEX, encoding="utf-8"))
    file_size = os.path.getsize(OUTPUT_TEX) / 1024
    print(f"\n[OK] Success! Output: {OUTPUT_TEX}")
    print(f"   Lines: {line_count}, Size: {file_size:.1f} KB")

    # Quick content verification
    with open(OUTPUT_TEX, "r", encoding="utf-8") as f:
        content = f.read()

    chapters = re.findall(r'\\(?:chapter|section)\{CHAPTER (\d+)', content)
    print(f"   Chapters found: {sorted(set(chapters), key=int)}")

    tables_count = content.count('\\begin{longtable}') + content.count('\\begin{tabular')
    figures_count = content.count('\\begin{figure}')
    equations_count = len(re.findall(r'\\\[.*?\\\]', content, re.DOTALL))
    print(f"   Tables: {tables_count}, Figures: {figures_count}, Display equations: {equations_count}")

if __name__ == "__main__":
    main()
