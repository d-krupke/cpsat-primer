"""
This file is used to build the `README.md` file from all markdown files in the current directory.
It also converts the markdown files to a format that can be used by the `mdbook` tool, which we
use to generate the website.
"""

import logging
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

def collect_anchors(content) -> list[str]:
    """
    Collect all anchors in a file an return their names as a list.
    This is important as we will have to replace the anchors by links in the mdBook version.
    We only use html anchors of the form `<a name="04-modelling-circuit"></a>` and no markdown anchors.
    """
    # ignoring the closing tag `</a>` as we only need the name.
    anchors = re.findall(r'<a\s+name\s*=\s*"(.*?)"\s*>', content)
    # find and warn about duplicates
    duplicates = [x for x in anchors if anchors.count(x) > 1]
    if duplicates:
        logging.warning(
            f"Found duplicate anchors in {file}: {', '.join(duplicates)}. Please check the file for errors."
        )
    # warn if any anchor has a bad name
    for anchor in anchors:
        if not re.match(r"^[a-zA-Z0-9_-]+$", anchor):
            logging.warning(
                f"Anchor {anchor} in {file} has a bad name. Please check the file for errors."
            )
    return anchors

def collect_anchor_links(content: str) -> list[str]:
    """
    Collect all anchor links in a file and return their names as a list.
    This is important as we will have to replace the anchors by links in the mdBook version.
    We only accept markdown links of the form `[text](#04-modelling-circuit)` and no html anchors.
    """
    # ignoring the closing tag `</a>` as we only need the name.
    anchors = re.findall(r'\[.*?\]\(#(.*?)\)', content)
    # warn if any anchor has a bad name
    for anchor in anchors:
        if not re.match(r"^[a-zA-Z0-9_-]+$", anchor):
            logging.warning(
                f"Anchor link {anchor} in {file} has a bad name. Please check the file for errors."
            )
    return anchors

class MarkdownFile:
    """
    A class to represent a markdown file.
    It has a name and a list of anchors.
    """

    def __init__(self, path: str, target: str):
        self.path = path
        self.target = target
        self.url = path.replace(".md", ".html")
        self.content = ""
        with open(path, "r") as f:
            self.content = f.read()
        self.anchors = collect_anchors(self.content)
        self.anchor_links = collect_anchor_links(self.content)

    def get_anchor_url(self, anchor: str) -> str:
        """
        Get the url for an anchor.
        The url is the path to the file + the anchor name.
        """
        if anchor not in self.anchors:
            raise ValueError(
                f"Anchor {anchor} not found in {self.path}. Available anchors: {self.anchors}"
            )
        return f"{self.url}#{anchor}"
    

class Document:
    def __init__(self, files: list[MarkdownFile]):
        self.files = files
        self.files_by_name = {file.path: file for file in files}
        self.anchors: dict[str, MarkdownFile] = {}
        for file in files:
            for anchor in file.anchors:
                if anchor in self.anchors:
                    raise ValueError(
                        f"Anchor {anchor} already exists in {self.anchors[anchor]} and {file.path}"
                    )
                self.anchors[anchor] = file
        
    def get_anchor_url(self, anchor: str) -> str:
        """
        Get the url for an anchor.
        The url is the path to the file + the anchor name.
        """
        if anchor not in self.anchors:
            raise ValueError(
                f"Anchor {anchor} not found. Available anchors: {self.anchors}"
            )
        return self.anchors[anchor].get_anchor_url(anchor)
    
    def write_readme(self, path: str):
        """
        Write the readme file.
        The readme file is a concatenation of all files in the document.
        """
        with open(path, "w") as f:
            f.write(
                "*A book-style version of this primer is available at [https://d-krupke.github.io/cpsat-primer/](https://d-krupke.github.io/cpsat-primer/).*\n\n"
            )
            disclaimer = "<!-- This file was generated by the `build.py` script. Do not edit it manually. -->\n"
            for file in self.files:
                print(f"Adding {file.path} to README.md")
                content = file.content
                f.write(disclaimer)
                f.write(f"<!-- {file.path} -->\n")
                f.write(convert_for_readme(content))
                f.write("\n\n")

    def _replace_anchors_by_urls(self, content: str, anchors: dict[str, str]) -> str:
        """
        Replace all anchors in the content with their urls.
        For this, all md links of the form `[text](#anchor)` are replaced with `[text](url)`.
        The url is the path to the file + the anchor name.
        Only the anchors that are in the anchors dict are replaced, as we do not want to replace
        anchors within the same file.
        """
        for anchor, url in anchors.items():
            # replace all links of the form `[text](#anchor)` with `[text](url)`
            content = re.sub(
                rf"\[([^\]]+)\]\(#({anchor})\)",
                rf"[\1]({url})",
                content,
            )
        return content

    def _get_mdbook_content(self, file: MarkdownFile) -> str:
        """
        Get the content of a file for mdbook. This will have multiple parts replaced.
        """
        anchors = {
            anchor: self.get_anchor_url(anchor)
            for anchor, f in self.anchors.items()
            if f != file  # skip anchors in the same file
        }
        content = file.content
        content = self._replace_anchors_by_urls(content, anchors)
        content = convert_for_mdbook(content)
        return content

    def write_mdbook(self, path: str):
        """
        Write the mdbook files.
        The mdbook files are a concatenation of all files in the document.
        """
        for file in self.files:
            content = self._get_mdbook_content(file)
            target_path = Path(path) / file.target
            print(f"Writing {file.path} to {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w") as book_file:
                book_file.write(content)


if __name__ == "__main__":
    # get all markdown files that start with a number
    # [f for f in os.listdir() if f.endswith(".md") and f[0].isdigit()]
    document = Document([
        MarkdownFile("00_intro.md", "00_intro.md"),
        MarkdownFile("01_installation.md", "01_installation.md"),
        MarkdownFile("02_example.md", "02_example.md"),
        MarkdownFile("04_modelling.md", "04_modelling.md"),
        MarkdownFile("04B_advanced_modelling.md", "04B_advanced_modelling.md"),
        MarkdownFile("05_parameters.md", "05_parameters.md"),
        MarkdownFile("understanding_the_log.md", "understanding_the_log.md"),
        MarkdownFile("07_under_the_hood.md", "07_under_the_hood.md"),
        MarkdownFile("03_big_picture.md", "03_big_picture.md"),
        MarkdownFile("06_coding_patterns.md", "06_coding_patterns.md"),
        MarkdownFile("building_an_optimization_api.md", "building_an_optimization_api.md"),
        MarkdownFile("chapters/machine_learning.md", "chapters/machine_learning.md"),
        MarkdownFile("08_benchmarking.md", "08_benchmarking.md"),
        MarkdownFile("09_lns.md", "09_lns.md"),
    ])
    document.write_readme("README.md")
    # write the mdbook files to the .mdbook directory
    document.write_mdbook("./.mdbook/")