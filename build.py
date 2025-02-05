"""
This file is used to build the `README.md` file from all markdown files in the current directory.
It also converts the markdown files to a format that can be used by the `mdbook` tool, which we
use to generate the website.
"""

import os
import re
from pathlib import Path


def _create_pretty_warning_box(msg):
    return f"""
<table style="width: 100%; border: 2px solid #ccc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
<tr>
<td style="padding: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="width: 10%;">
    <img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/warning_platypus.webp" alt="Description of image" style="width: 100%;">
  </div>
  <div style="width: 90%;">

{msg}

  </div>
</div>
    </td>
  </tr>
</table>
    """


def _create_tip_box(msg):
    return f"""
<table style="width: 100%; border: 2px solid #ccc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
<tr>
<td style="padding: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="width: 10%;">
    <img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/idea_platypus.webp" alt="Description of image" style="width: 100%;">
  </div>
  <div style="width: 90%;">

{msg}

  </div>
</div>
    </td>
  </tr>
</table>
    """


def _create_info_box(msg):
    return f"""
<table style="width: 100%; border: 2px solid #ccc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
<tr>
<td style="padding: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="width: 10%;">
                    <img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/info_platypus.webp" alt="Description of image" style="width: 100%;">
                </div>
                <div style="width: 90%;">

{msg}

  </div>
</div>
    </td>
  </tr>
</table>
    """


def _create_reference_box(msg):
    return f"""
<table style="width: 100%; border: 2px solid #ccc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
<tr>
<td style="padding: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="width: 10%;">
                    <img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/book_platypus.webp" alt="Description of image" style="width: 100%;">
                </div>
                <div style="width: 90%;">

{msg}

  </div>
</div>
    </td>
  </tr>
</table>
    """


def _create_video_box(msg):
    return f"""
<table style="width: 100%; border: 2px solid #ccc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
<tr>
<td style="padding: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center;">

<div style="width: 85%;">
{msg}
</div>
<div style="width: 15%; padding-right: 10px;">
<img src="https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/tv_platypus.webp" alt="Description of image" style="width: 100%; border-radius: 4px;">
</div>
</div>
</td>
</tr>
</table>
    """


def replace_video_boxes(content):
    """
    A video box starts with `> :video:` and ends with a line that does not start with `>`.
    """
    lines = content.split("\n")
    new_content = ""
    collect_video = False
    video_msg = ""
    for line in lines:
        if line.startswith("> :video:"):
            collect_video = True
            video_msg += line[len("> :video:") :] + "\n"
        elif collect_video:
            if line == ">":
                continue
            if line.startswith("> "):
                video_msg += line[len("> ") :] + "\n"
            else:
                new_content += _create_video_box(video_msg)
                new_content += "\n"
                collect_video = False
                video_msg = ""
                new_content += line + "\n"
        else:
            new_content += line + "\n"
    return new_content


def replace_reference_boxes(content):
    """
    A reference box starts with `> :reference:` and ends with a line that does not start with `>`.
    For github markdown, it just converts to an info box.
    """
    lines = content.split("\n")
    new_content = ""
    collect_reference = False
    reference_msg = ""
    for line in lines:
        if line.startswith("> :reference:"):
            collect_reference = True
            reference_msg += line[len("> :reference:") :] + "\n"
        elif collect_reference:
            if line == ">":
                continue
            if line.startswith("> "):
                reference_msg += line[len("> ") :] + "\n"
            else:
                new_content += _create_reference_box(reference_msg)
                new_content += "\n"
                collect_reference = False
                reference_msg = ""
                new_content += line + "\n"
        else:
            new_content += line + "\n"
    return new_content


def replace_warning_boxes(content):
    """
    A warning box starts with `> :warning:` and ends with a line that does not start with `>`.
    """
    lines = content.split("\n")
    new_content = ""
    collect_warning = False
    warning_msg = ""
    for line in lines:
        if line.startswith("> :warning:"):
            collect_warning = True
            warning_msg += line[len("> :warning:") :] + "\n"
        elif line.startswith("> [!WARNING]"):
            collect_warning = True
            warning_msg += line[len("> [!WARNING]") :] + "\n"
        elif collect_warning:
            if line == ">":
                continue
            if line.startswith("> "):
                warning_msg += line[len("> ") :] + "\n"
            else:
                new_content += _create_pretty_warning_box(warning_msg)
                new_content += "\n"
                collect_warning = False
                warning_msg = ""
                new_content += line + "\n"
        else:
            new_content += line + "\n"
    return new_content


def replace_tip_boxes(content):
    """
    A tip box starts with `> [!TIP]` and ends with a line that does not start with `>`.
    """
    lines = content.split("\n")
    new_content = ""
    collect_tip = False
    tip_msg = ""
    for line in lines:
        if line.startswith("> [!TIP]"):
            collect_tip = True
            tip_msg += line[len("> [!TIP]") :] + "\n"
        elif collect_tip:
            if line == ">":
                continue
            if line.startswith("> "):
                tip_msg += line[len("> ") :] + "\n"

            else:
                new_content += _create_tip_box(tip_msg)
                new_content += "\n"
                collect_tip = False
                tip_msg = ""
                new_content += line + "\n"
        else:
            new_content += line + "\n"
    return new_content


def replace_info_boxes(content):
    """
    An info box starts with `> [!NOTE]` and ends with a line that does not start with `>`.
    """
    lines = content.split("\n")
    new_content = ""
    collect_info = False
    info_msg = ""
    for line in lines:
        if line.startswith("> [!NOTE]"):
            collect_info = True
            info_msg += line[len("> [!NOTE]") :] + "\n"
        elif collect_info:
            if line == ">":
                continue
            if line.startswith("> "):
                info_msg += line[len("> ") :] + "\n"

            else:
                new_content += _create_info_box(info_msg)
                new_content += "\n"
                collect_info = False
                info_msg = ""
                new_content += line + "\n"
        else:
            new_content += line + "\n"
    return new_content


def convert_for_mdbook(content):
    footer = """
---
*The CP-SAT Primer is authored by [Dominik Krupke](https://github.com/d-krupke) at [TU Braunschweig, Algorithms Division](https://www.ibr.cs.tu-bs.de/alg/index.html). It is licensed under the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/).*

*The primer is written for educational purposes and may be incomplete or incorrect in places. If you find this primer helpful, please star the [GitHub repository](https://github.com/d-krupke/cpsat-primer/), to let me know with a single click that it has made an impact. As an academic, I also enjoy hearing about how you use CP-SAT to solve real-world problems.*
    """
    content = (
        "<!-- This file was generated by the `build.py` script. Do not edit it manually. -->\n"
        + content
    )
    # replace all inline math `$...$` with `\\( ... \\)` using regex.
    # always use the smallest possible match for the `...` part.
    content = re.sub(r"\$(.*?)\$", r"\\\\( \1 \\\\)", content)
    # replace all math modes "```math ... ```" with `\\[ ... \\]` using regex.
    # always use the smallest possible match for the `...` part.
    content = re.sub(r"```math(.*?)```", r"\\\\[ \1 \\\\]", content, flags=re.DOTALL)
    content = replace_warning_boxes(content)
    content = replace_tip_boxes(content)
    content = replace_info_boxes(content)
    content = replace_reference_boxes(content)
    content = replace_video_boxes(content)
    # replace all `:warning:` with the unicode character for a warning sign.
    content = content.replace(":warning:", "⚠️")

    # replace all anchor links `(#01-installation)` by `(./01_installation.md)`.
    # you have to replace the `#` with `./` and `-` with `_`, and attach `.md` at the end.
    def replace_relative(match):
        md_path = match.group(1).replace("-", "_") + ".md"
        all_md_files = [f for f in os.listdir() if f.endswith(".md")]
        for file in all_md_files:
            if file.endswith(md_path):
                md_path = file
                break
        return f"(./{md_path})" if Path(md_path).exists() else f"(#{match.group(1)})"

    explicit_replacements = {
        "#chapters-machine-learning": "./chapters/machine_learning.md",
    }
    for key, value in explicit_replacements.items():
        content = content.replace(key, value)

    content = re.sub(
        r"\(#(.*?)\)",
        replace_relative,
        content,
    )
    # replace in all links that lead to a .png file the `github.com` with `raw.githubusercontent.com`.
    content = re.sub(
        r"\((.*?\.png)\)",
        lambda match: match.group(0).replace(
            "https://github.com/d-krupke/cpsat-primer/blob/main/",
            "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/",
        ),
        content,
    )
    content = re.sub(
        r"\((.*?\.gif)\)",
        lambda match: match.group(0).replace(
            "https://github.com/d-krupke/cpsat-primer/blob/main/",
            "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/",
        ),
        content,
    )
    content = re.sub(
        r"\((.*?\.jpg)\)",
        lambda match: match.group(0).replace(
            "https://github.com/d-krupke/cpsat-primer/blob/main/",
            "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/",
        ),
        content,
    )
    content = re.sub(
        r"\((.*?\.webp)\)",
        lambda match: match.group(0).replace(
            "https://github.com/d-krupke/cpsat-primer/blob/main/",
            "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/",
        ),
        content,
    )

    content += footer
    return content


def convert_for_readme(content: str) -> str:
    # If we have a `<!-- START_SKIP_FOR_README -->` and `<!-- STOP_SKIP_FOR_README -->` in the content, we skip this part.
    # These will be single lines, so we can use `splitlines()` to get the lines and then check if we should skip after having
    # removed all whitespaces which can be created by the formatting.
    lines = content.splitlines()
    skip = False
    new_lines = []
    for line in lines:
        if "<!-- START_SKIP_FOR_README -->" in line:
            skip = True
        if not skip:
            new_lines.append(line)
        if "<!-- STOP_SKIP_FOR_README -->" in line:
            skip = False
    content = "\n".join(new_lines)
    # replace all `> :reference:` with `> [!NOTE]`.
    content = content.replace("> :reference:", "> [!NOTE]")
    content = content.replace("> :video:", "> [!TIP]")
    return content


if __name__ == "__main__":
    # get all markdown files that start with a number
    # [f for f in os.listdir() if f.endswith(".md") and f[0].isdigit()]
    markdown_files = [
        "00_intro.md",
        "01_installation.md",
        "02_example.md",
        "04_modelling.md",
        "04B_advanced_modelling.md",
        "05_parameters.md",
        "understanding_the_log.md",
        "07_under_the_hood.md",
        "03_big_picture.md",
        "06_coding_patterns.md",
        "building_an_optimization_api.md",
        "chapters/machine_learning.md",
        "08_benchmarking.md",
        "09_lns.md",
    ]

    # concat them and write them to `README.md`
    with open("README.md", "w") as f:
        f.write(
            "*A book-style version of this primer is available at [https://d-krupke.github.io/cpsat-primer/](https://d-krupke.github.io/cpsat-primer/).*\n\n"
        )
        disclaimer = "<!-- This file was generated by the `build.py` script. Do not edit it manually. -->\n"
        for file in markdown_files:
            print(f"Adding {file} to README.md")
            with open(file, "r") as current_file:
                content = current_file.read()
                f.write(disclaimer)
                f.write(f"<!-- {file} -->\n")
                f.write(convert_for_readme(content))
                f.write("\n\n")
                (Path("./.mdbook/") / file).parent.mkdir(parents=True, exist_ok=True)
                with open(Path("./.mdbook/") / file, "w") as book_file:
                    book_file.write(convert_for_mdbook(content))
