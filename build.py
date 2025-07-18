"""
This file is used to build the `README.md` file from all markdown files in the current directory.
It also converts the markdown files to a format that can be used by the `mdbook` tool, which we
use to generate the website.
"""

import logging
import re
from pathlib import Path
from typing import List, Callable

# Configuration for each box type
BOX_CONFIG = {
    "warning": {
        "tokens": ["> :warning:", "> [!WARNING]"],
        "icon": "warning_platypus.webp",
        "img_width": "10%",
        "content_width": "90%",
        "img_style": "",
        "reverse": False,
    },
    "tip": {
        "tokens": ["> [!TIP]"],
        "icon": "idea_platypus.webp",
        "img_width": "10%",
        "content_width": "90%",
        "img_style": "",
        "reverse": False,
    },
    "info": {
        "tokens": ["> [!NOTE]"],
        "icon": "info_platypus.webp",
        "img_width": "10%",
        "content_width": "90%",
        "img_style": "",
        "reverse": False,
    },
    "reference": {
        "tokens": ["> :reference:"],
        "icon": "book_platypus.webp",
        "img_width": "10%",
        "content_width": "90%",
        "img_style": "",
        "reverse": False,
    },
    "video": {
        "tokens": ["> :video:"],
        "icon": "tv_platypus.webp",
        "img_width": "15%",
        "content_width": "85%",
        "img_style": "padding-right: 10px; border-radius: 4px;",
        "reverse": True,
    },
}
BASE_URL = "https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/"
COMMON_TABLE_STYLE = (
    "width: 100%; border: 2px solid #ccc; border-radius: 8px; "
    "box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);"
)


def _create_box(msg: str, config: dict) -> str:
    """
    Render a generic HTML box.
    """
    icon_url = BASE_URL + config["icon"]
    # build image and content divs
    img_div = (
        f'<div style="width: {config["img_width"]}; {config["img_style"]}">'
        f'<img src="{icon_url}" alt="{config["icon"]}" style="width: 100%;">'
        f"</div>"
    )
    content_div = f'<div style="width: {config["content_width"]};">\n\n{msg}\n</div>'
    # reverse order if needed (video puts content first)
    left, right = (
        (img_div, content_div) if not config["reverse"] else (content_div, img_div)
    )

    return f"""
<table style="{COMMON_TABLE_STYLE}">
  <tr>
    <td style="padding: 10px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        {left}
        {right}
      </div>
    </td>
  </tr>
</table>
"""


def _replace_boxes(
    content: str, tokens: List[str], create_func: Callable[[str], str]
) -> str:
    """
    Generic box replacement logic: collects lines starting with any of the given tokens
    until a non-quoted line appears, then uses create_func to render.
    """
    lines = content.splitlines(keepends=True)
    out, collecting, buffer = [], False, []
    for line in lines:
        if not collecting and any(line.startswith(tok) for tok in tokens):
            collecting = True
            # strip the matched token
            for tok in tokens:
                if line.startswith(tok):
                    buffer.append(line[len(tok) :].lstrip())
                    break
        elif collecting:
            if line.strip() == ">":
                continue
            if line.startswith("> "):
                buffer.append(line[2:])
            else:
                # end of box block
                msg = "".join(buffer)
                out.append(create_func(msg))
                out.append("\n")
                collecting = False
                buffer = []
                out.append(line)
        else:
            out.append(line)
    # handle case where content ends while collecting
    if collecting and buffer:
        out.append(create_func("".join(buffer)))
    return "".join(out)


# Specific replace functions


def replace_warning_boxes(content: str) -> str:
    return _replace_boxes(
        content,
        BOX_CONFIG["warning"]["tokens"],
        lambda msg: _create_box(msg, BOX_CONFIG["warning"]),
    )


def replace_tip_boxes(content: str) -> str:
    return _replace_boxes(
        content,
        BOX_CONFIG["tip"]["tokens"],
        lambda msg: _create_box(msg, BOX_CONFIG["tip"]),
    )


def replace_info_boxes(content: str) -> str:
    return _replace_boxes(
        content,
        BOX_CONFIG["info"]["tokens"],
        lambda msg: _create_box(msg, BOX_CONFIG["info"]),
    )


def replace_reference_boxes(content: str) -> str:
    return _replace_boxes(
        content,
        BOX_CONFIG["reference"]["tokens"],
        lambda msg: _create_box(msg, BOX_CONFIG["reference"]),
    )


def replace_video_boxes(content: str) -> str:
    return _replace_boxes(
        content,
        BOX_CONFIG["video"]["tokens"],
        lambda msg: _create_box(msg, BOX_CONFIG["video"]),
    )


def convert_for_mdbook(content):
    footer = """
---
*The **CP-SAT Primer** is maintained by [Dominik Krupke](https://github.com/d-krupke) at the [Algorithms Division, TU Braunschweig](https://www.ibr.cs.tu-bs.de/alg/index.html), and is licensed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). [Contributions are welcome](https://github.com/d-krupke/cpsat-primer/pulls).*

*If you find the primer helpful, consider leaving a ⭐ on [GitHub (522⭐)](https://github.com/d-krupke/cpsat-primer) or [sharing your feedback/experience](https://www.linkedin.com/in/dominik-krupke-9869b2241/). Your support helps improve and sustain this free resource.*
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
            f"Found duplicate anchors: {', '.join(duplicates)}. Please check the file for errors."
        )
    # warn if any anchor has a bad name
    for anchor in anchors:
        if not re.match(r"^[a-zA-Z0-9_-]+$", anchor):
            logging.warning(
                f"Anchor {anchor} has a bad name. Please check the file for errors."
            )
    return anchors


def collect_anchor_links(content: str) -> list[str]:
    """
    Collect all anchor links in a file and return their names as a list.
    This is important as we will have to replace the anchors by links in the mdBook version.
    We only accept markdown links of the form `[text](#04-modelling-circuit)` and no html anchors.
    """
    # ignoring the closing tag `</a>` as we only need the name.
    anchors = re.findall(r"\[.*?\]\(#(.*?)\)", content)
    # warn if any anchor has a bad name
    for anchor in anchors:
        if not re.match(r"^[a-zA-Z0-9_-]+$", anchor):
            logging.warning(
                f"Anchor link {anchor} has a bad name. Please check the file for errors."
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
        self.url = target.replace(".md", ".html")
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
    document = Document(
        [
            MarkdownFile("./chapters/00_intro.md", "00_intro.md"),
            MarkdownFile("./chapters/01_installation.md", "01_installation.md"),
            MarkdownFile("./chapters/02_example.md", "02_example.md"),
            MarkdownFile("./chapters/04_modelling.md", "04_modelling.md"),
            MarkdownFile(
                "./chapters/04B_advanced_modelling.md", "04B_advanced_modelling.md"
            ),
            MarkdownFile("./chapters/05_parameters.md", "05_parameters.md"),
            MarkdownFile("understanding_the_log.md", "understanding_the_log.md"),
            MarkdownFile("./chapters/07_under_the_hood.md", "07_under_the_hood.md"),
            MarkdownFile("./chapters/03_big_picture.md", "03_big_picture.md"),
            MarkdownFile("./chapters/06_coding_patterns.md", "06_coding_patterns.md"),
            MarkdownFile(
                "./chapters/test_driven_optimization.md", "test_driven_optimization.md"
            ),
            MarkdownFile(
                "./chapters/building_an_optimization_api.md",
                "building_an_optimization_api.md",
            ),
            MarkdownFile(
                "chapters/machine_learning.md", "chapters/machine_learning.md"
            ),
            MarkdownFile("./chapters/08_benchmarking.md", "08_benchmarking.md"),
            MarkdownFile("./chapters/09_lns.md", "09_lns.md"),
        ]
    )
    document.write_readme("README.md")
    # write the mdbook files to the .mdbook directory
    document.write_mdbook("./.mdbook/")
