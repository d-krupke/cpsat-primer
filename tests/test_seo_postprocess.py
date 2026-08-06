"""
Tests for `seo_postprocess.py`, the post-`mdbook build` step that adds canonical
link tags and a sitemap to the generated site.

Created 2026-08-06. The behavior under test is what Google reacts to, and a
regression here is invisible locally and only shows up weeks later in Search
Console, so the rules are pinned explicitly rather than eyeballed after a deploy.

The book directory is built by a small helper instead of a fixture so each test
states the exact page set it cares about.
"""

from pathlib import Path

import seo_postprocess as sp

BASE_URL = "https://example.org/book/"


def page(title: str = "A chapter", body: str = "content") -> str:
    """A minimal stand-in for an mdBook chapter page."""
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n  <head>\n"
        f"    <title>{title}</title>\n"
        "  </head>\n"
        f"  <body>{body}</body>\n</html>\n"
    )


def redirect_stub(target: str) -> str:
    """A stand-in for the pages mdBook writes for `[output.html.redirect]`."""
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n  <head>\n"
        f'    <meta http-equiv="refresh" content="0; URL={target}">\n'
        f'    <link rel="canonical" href="{target}">\n'
        "  </head>\n  <body></body>\n</html>\n"
    )


def build_book(root: Path, pages: dict[str, str]) -> Path:
    """Write `pages` (book-relative path -> html) into a fresh book directory."""
    book_dir = root / "book"
    for rel, html in pages.items():
        target = book_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return book_dir


# --- URL mapping -------------------------------------------------------------


def test_index_maps_to_directory_url():
    """`index.html` must canonicalize to the directory URL, since that is the
    form users, GitHub and existing backlinks point at."""
    assert sp.page_url(BASE_URL, sp.PurePosixPath("index.html")) == BASE_URL
    assert (
        sp.page_url(BASE_URL, sp.PurePosixPath("sub/index.html"))
        == "https://example.org/book/sub/"
    )


def test_chapter_maps_to_its_own_url():
    """Ordinary chapters get a self-referencing URL under the base URL."""
    assert (
        sp.page_url(BASE_URL, sp.PurePosixPath("intro.html"))
        == "https://example.org/book/intro.html"
    )


def test_base_url_trailing_slash_is_normalized():
    """A base URL given without a trailing slash must not produce `book intro.html`
    glued together without a separator."""
    assert (
        sp.page_url("https://example.org/book", sp.PurePosixPath("intro.html"))
        == "https://example.org/book/intro.html"
    )


# --- canonical injection -----------------------------------------------------


def test_canonical_is_inserted_into_head():
    html = sp.insert_canonical(page(), "https://example.org/book/intro.html")
    assert '<link rel="canonical" href="https://example.org/book/intro.html">' in html
    # It has to sit inside <head>, not after it, or browsers/crawlers ignore it.
    assert html.index('rel="canonical"') < html.index("</head>")


def test_canonical_injection_is_idempotent():
    """The deploy may run the script twice; a second canonical tag would be a
    conflicting signal, so an existing one is left untouched."""
    once = sp.insert_canonical(page(), "https://example.org/book/intro.html")
    twice = sp.insert_canonical(once, "https://example.org/book/other.html")
    assert once == twice
    assert twice.count('rel="canonical"') == 1


def test_page_without_head_is_left_alone():
    """Defensive: a page we cannot anchor the tag in is returned unchanged rather
    than corrupted."""
    assert sp.insert_canonical("<html><body>x</body></html>", "u") == (
        "<html><body>x</body></html>"
    )


# --- page classification -----------------------------------------------------


def test_redirect_stub_is_recognized():
    assert sp.is_redirect_stub(redirect_stub("intro.html"))
    assert not sp.is_redirect_stub(page())


def test_non_document_html_is_not_a_page():
    """`src = "."` makes mdBook copy Google's site-verification token through to
    the output. It ends in `.html` but is one line of text, and must not be
    treated as a page."""
    assert not sp.is_rendered_page("google-site-verification: googlefoo.html\n")
    assert sp.is_rendered_page(page())


def test_noindex_page_is_recognized():
    """mdBook marks print.html noindex; anything so marked stays out of the sitemap."""
    noindex = page().replace(
        "<head>", '<head>\n    <meta name="robots" content="noindex">'
    )
    assert sp.is_noindex(noindex)
    assert not sp.is_noindex(page())


# --- end-to-end --------------------------------------------------------------


def test_postprocess_writes_canonicals_and_sitemap(tmp_path):
    """The happy path: real chapters get self-canonicals and sitemap entries."""
    book = build_book(
        tmp_path,
        {
            "index.html": page("Intro", "intro body"),
            "intro.html": page("Intro", "intro body"),
            "modelling.html": page("Modelling", "modelling body"),
        },
    )

    urls = sp.postprocess(book, BASE_URL)

    modelling = (book / "modelling.html").read_text()
    assert (
        '<link rel="canonical" href="https://example.org/book/modelling.html">'
        in modelling
    )
    assert "https://example.org/book/modelling.html" in urls

    sitemap = (book / "sitemap.xml").read_text()
    assert "<loc>https://example.org/book/modelling.html</loc>" in sitemap
    assert sitemap.startswith("<?xml")


def test_index_duplicate_is_consolidated_to_the_root(tmp_path):
    """This is the whole point of the script: mdBook copies the first chapter to
    index.html byte for byte, which is what Search Console flagged as
    "Duplicate without user-selected canonical". Both copies must name the site
    root as canonical, and the duplicate must not appear twice in the sitemap."""
    identical = page("Intro", "intro body")
    book = build_book(tmp_path, {"index.html": identical, "intro.html": identical})

    urls = sp.postprocess(book, BASE_URL)

    assert (
        f'<link rel="canonical" href="{BASE_URL}">' in (book / "intro.html").read_text()
    )
    assert (
        f'<link rel="canonical" href="{BASE_URL}">' in (book / "index.html").read_text()
    )
    assert urls == [BASE_URL]


def test_distinct_intro_keeps_its_own_canonical(tmp_path):
    """If the first chapter and index.html ever stop being identical, intro.html
    is a page in its own right and must not be pointed at the root."""
    book = build_book(
        tmp_path,
        {
            "index.html": page("Home", "landing body"),
            "intro.html": page("Intro", "intro body"),
        },
    )

    urls = sp.postprocess(book, BASE_URL)

    assert (
        '<link rel="canonical" href="https://example.org/book/intro.html">'
        in (book / "intro.html").read_text()
    )
    assert sorted(urls) == [BASE_URL, "https://example.org/book/intro.html"]


def test_redirects_print_and_404_stay_out(tmp_path):
    """Redirect stubs keep the canonical mdBook gave them, and none of the
    non-indexable pages may leak into the sitemap and invite a crawl."""
    noindex = page("Print").replace(
        "<head>", '<head>\n    <meta name="robots" content="noindex">'
    )
    book = build_book(
        tmp_path,
        {
            "index.html": page("Intro", "intro body"),
            "00_intro.html": redirect_stub("intro.html"),
            "print.html": noindex,
            "404.html": page("Not found"),
            "googlefoo.html": "google-site-verification: googlefoo.html\n",
        },
    )

    urls = sp.postprocess(book, BASE_URL)

    assert urls == [BASE_URL]
    stub = (book / "00_intro.html").read_text()
    assert stub.count('rel="canonical"') == 1
    assert 'href="intro.html"' in stub
    # The verification token must survive byte for byte or Google unverifies the site.
    assert (
        book / "googlefoo.html"
    ).read_text() == "google-site-verification: googlefoo.html\n"
    sitemap = (book / "sitemap.xml").read_text()
    for excluded in ("00_intro.html", "print.html", "404.html", "googlefoo.html"):
        assert excluded not in sitemap


def test_robots_points_at_the_sitemap(tmp_path):
    book = build_book(tmp_path, {"index.html": page()})

    sp.postprocess(book, BASE_URL)

    robots = (book / "robots.txt").read_text()
    assert "Sitemap: https://example.org/book/sitemap.xml" in robots


def test_rerunning_does_not_change_the_output(tmp_path):
    """The deploy is not guaranteed to run on a clean tree; a second pass must be
    a no-op rather than accumulating tags or sitemap entries."""
    book = build_book(
        tmp_path,
        {
            "index.html": page("Intro", "intro body"),
            "modelling.html": page("Modelling"),
        },
    )

    first_urls = sp.postprocess(book, BASE_URL)
    snapshot = {p.name: p.read_text() for p in book.rglob("*") if p.is_file()}
    second_urls = sp.postprocess(book, BASE_URL)

    assert first_urls == second_urls
    assert snapshot == {p.name: p.read_text() for p in book.rglob("*") if p.is_file()}
