#!/usr/bin/env python3
"""Build docs-site/content from README.md and docs/*.md, the docs' only source.

Each page gets Zola front matter (title, navigation, previous and next, its
source path), and links between the Markdown files become Zola internal links,
which the build's link checker verifies.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs-site"
# Navigation order: the overview, then the guides in reading order.
PAGES = [
    ("README.md", ""),
    ("docs/cli.md", "cli/"),
    ("docs/profiles.md", "profiles/"),
    ("docs/unattended.md", "unattended/"),
    ("docs/harnesses.md", "harnesses/"),
    ("docs/token-usage.md", "token-usage/"),
    ("docs/integration.md", "integration/"),
    ("docs/glossary.md", "glossary/"),
]
# Links to repository files that are not pages, such as CONTRIBUTING.md.
REPO_FILE = re.compile(r"\]\(([A-Z][A-Za-z_-]*\.md)\)")
GITHUB = "https://github.com/wakamex/agent-orchestration-process/blob/main/"
LINK = re.compile(r"\]\((?:docs/)?([a-z-]+)\.md(#[^)]*)?\)")


def title_and_body(text: str) -> tuple[str, str]:
    first, _, rest = text.partition("\n")
    if not first.startswith("# "):
        raise SystemExit("each page must start with a level-one heading")
    return first[2:].strip(), rest.lstrip("\n")


def toml_string(value: str) -> str:
    return json.dumps(value)


def main() -> None:
    content = SITE / "content"
    shutil.rmtree(content, ignore_errors=True)
    content.mkdir()
    pages = []
    for source, path in PAGES:
        title, body = title_and_body((ROOT / source).read_text())
        pages.append((source, path, title, body))
    navigation = {"groups": [{"id": "aop", "title": "AOP", "items": []}]}
    for index, (source, path, title, body) in enumerate(pages):
        navigation["groups"][0]["items"].append(
            {"title": "Overview" if path == "" else title, "path": path}
        )
        extra = [f"source_path = {toml_string(source)}", 'navigation_group = "aop"']
        for key, neighbour in (("previous", index - 1), ("next", index + 1)):
            if 0 <= neighbour < len(pages):
                _, other_path, other_title, _ = pages[neighbour]
                other_title = "Overview" if other_path == "" else other_title
                extra.append(
                    f"{key} = {{ title = {toml_string(other_title)}, path = {toml_string(other_path)} }}"
                )
        body = REPO_FILE.sub(lambda m: f"]({GITHUB}{m.group(1)})", body)
        body = LINK.sub(lambda m: f"](@/{m.group(1)}.md{m.group(2) or ''})", body)
        if path == "":
            front = ["+++", f"title = {toml_string(title)}", 'sort_by = "none"', "", "[extra]", "toc = false", *extra, "+++"]
            (content / "_index.md").write_text("\n".join(front) + f"\n# {title}\n\n" + body)
        else:
            front = ["+++", f"title = {toml_string(title)}", "", "[extra]", *extra, "+++"]
            (content / f"{path.rstrip('/')}.md").write_text("\n".join(front) + "\n" + body)
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "navigation.json").write_text(json.dumps(navigation, indent=2) + "\n")


if __name__ == "__main__":
    main()
