import httpx
import pytest

from evals.crawl.discover import (
    CrawlResult,
    Site,
    campus_for,
    classify_link,
    crawl_site,
    own_hosts,
    parse_links,
    slugify,
)
from evals.crawl.http import DisallowedUrlError, PoliteClient, is_allowed_host, parse_robots


def test_allowed_hosts_are_iuh_only():
    assert is_allowed_host("iuh.edu.vn")
    assert is_allowed_host("pdt.iuh.edu.vn")
    assert is_allowed_host("PDT.IUH.EDU.VN:443")
    assert not is_allowed_host("iuh.edu.vn.evil.com")
    assert not is_allowed_host("notiuh.edu.vn")
    assert not is_allowed_host("drive.google.com")


def test_robots_html_redirect_is_treated_as_missing():
    parser = parse_robots(200, "text/html; charset=utf-8", "<html>Disallow: /</html>")
    assert parser.can_fetch("UniSageEvalBot", "https://iuh.edu.vn/anything")


def test_robots_rules_are_honoured():
    parser = parse_robots(200, "text/plain", "User-agent: *\nDisallow: /private/\n")
    assert parser.can_fetch("UniSageEvalBot", "https://x.iuh.edu.vn/public/a.pdf")
    assert not parser.can_fetch("UniSageEvalBot", "https://x.iuh.edu.vn/private/a.pdf")


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://pdt.iuh.edu.vn/Uploads/Quy che.PDF", "pdf"),
        ("https://pdt.iuh.edu.vn/Uploads/a%20b.pdf", "pdf"),
        ("https://fit.iuh.edu.vn/upload/x.docx", "other_doc"),
        ("https://drive.google.com/file/d/abc/view", "drive"),
        ("https://example.com/a.pdf", "skip"),
        ("https://pdt.iuh.edu.vn/logo.png", "skip"),
        ("https://pdt.iuh.edu.vn/Language/Change?lang=en", "skip"),
        ("https://sv.iuh.edu.vn/sinh-vien-dang-nhap.html", "skip"),
        ("https://pdt.iuh.edu.vn/danh-sach/thong-bao", "page"),
    ],
)
def test_classify_link(url, kind):
    assert classify_link(url) == kind


def test_parse_links_resolves_base_and_drops_noise():
    html = """
    <html><head><base href="https://iuh.edu.vn/" /></head><body>
      <a href="vi/phong-dao-tao.html">Phòng Đào tạo</a>
      <a href="vi/phong-dao-tao.html#top">dup</a>
      <a href="javascript:void(0)">js</a>
      <a href="mailto:a@b.c">mail</a>
      <iframe src="/Uploads/doc.pdf"></iframe>
    </body></html>
    """
    links = parse_links(html, "https://iuh.edu.vn/vi/cac-don-vi-truc-thuoc.html")
    assert [link.url for link in links] == [
        "https://iuh.edu.vn/vi/phong-dao-tao.html",
        "https://iuh.edu.vn/Uploads/doc.pdf",
    ]
    assert links[0].text == "Phòng Đào tạo"


def test_own_hosts_removes_shared_footer_and_portals():
    pages = {
        "fit": {"fit.iuh.edu.vn", "pdt.iuh.edu.vn", "lms.iuh.edu.vn"},
        "ptckt": {"ptckt.iuh.edu.vn", "pdt.iuh.edu.vn", "lms.iuh.edu.vn"},
        "pdt": {"pdt.iuh.edu.vn", "lms.iuh.edu.vn"},
        "other": {"pdt.iuh.edu.vn", "lms.iuh.edu.vn", "sv.iuh.edu.vn"},
    }
    assert own_hosts(pages) == {
        "fit": {"fit.iuh.edu.vn"},
        "ptckt": {"ptckt.iuh.edu.vn"},
        "pdt": set(),
        "other": set(),
    }


def test_slugify_and_campus():
    assert slugify("Phòng Đào tạo") == "phong-dao-tao"
    assert slugify("Khoa Công nghệ Nhiệt - Lạnh") == "khoa-cong-nghe-nhiet-lanh"
    assert campus_for("phan-hieu-quang-ngai", "qn.iuh.edu.vn") == "Quảng Ngãi"
    assert campus_for("co-so-thanh-hoa", "x.iuh.edu.vn") == "Thanh Hóa"
    assert campus_for("khoa-cong-nghe-thong-tin", "fit.iuh.edu.vn") == "HCM"


PAGES = {
    "/": '<a href="/list">list</a><a href="/a.pdf">A</a><a href="https://drive.google.com/x">d</a>',
    "/list": '<a href="/detail">detail</a><a href="/a.pdf">A again</a><a href="/f.docx">f</a>',
    "/detail": '<a href="/b.pdf">B</a><a href="https://other.iuh.edu.vn/c.pdf">C</a>',
}


def _site_transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    body = PAGES.get(request.url.path)
    if body is None:
        return httpx.Response(404)
    return httpx.Response(200, headers={"content-type": "text/html"}, text=body)


@pytest.mark.asyncio
async def test_crawl_site_collects_pdfs_once_and_respects_depth():
    site = Site("https://x.iuh.edu.vn/", "x.iuh.edu.vn", "Khoa X", "HCM")
    result = CrawlResult()
    async with PoliteClient(
        min_interval=0, transport=httpx.MockTransport(_site_transport)
    ) as client:
        await crawl_site(client, site, result, max_pages=10, max_depth=2)

    assert sorted(result.pdfs) == [
        "https://other.iuh.edu.vn/c.pdf",
        "https://x.iuh.edu.vn/a.pdf",
        "https://x.iuh.edu.vn/b.pdf",
    ]
    assert result.pdfs["https://x.iuh.edu.vn/b.pdf"].source_page == "https://x.iuh.edu.vn/detail"
    assert (site.pages_crawled, site.pdf_count, site.other_doc_count, site.drive_count) == (
        3,
        3,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_crawl_site_stops_at_max_depth():
    site = Site("https://x.iuh.edu.vn/", "x.iuh.edu.vn", "Khoa X", "HCM")
    result = CrawlResult()
    async with PoliteClient(
        min_interval=0, transport=httpx.MockTransport(_site_transport)
    ) as client:
        await crawl_site(client, site, result, max_pages=10, max_depth=1)

    assert site.pages_crawled == 2
    assert "https://x.iuh.edu.vn/b.pdf" not in result.pdfs


@pytest.mark.asyncio
async def test_client_refuses_non_iuh_and_robots_blocked_urls():
    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="User-agent: *\nDisallow: /secret\n",
            )
        return httpx.Response(200, text="ok")

    async with PoliteClient(min_interval=0, transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(DisallowedUrlError):
            await client.get("https://example.com/a.pdf")
        with pytest.raises(DisallowedUrlError):
            await client.get("https://x.iuh.edu.vn/secret/a.pdf")
        assert (await client.get("https://x.iuh.edu.vn/open")).text == "ok"
