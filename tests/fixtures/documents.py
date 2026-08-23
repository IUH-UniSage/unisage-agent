import io
import zipfile

import openpyxl
import pymupdf


def make_xlsx_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    """Build a minimal in-memory `.xlsx` with one header row and data rows."""

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def make_pdf_bytes(text: str) -> bytes:
    """Build a minimal in-memory single-page PDF containing `text`."""

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data: bytes = document.tobytes()
    document.close()
    return data


def make_pdf_bytes_with_table(intro_text: str, outro_text: str) -> bytes:
    """Build a PDF with an intro paragraph, a 2-column/3-row grid table, then an outro paragraph."""

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), intro_text)

    x0, y0, cell_w, cell_h, rows, cols = 72, 120, 80, 20, 3, 2
    for row in range(rows + 1):
        page.draw_line((x0, y0 + row * cell_h), (x0 + cols * cell_w, y0 + row * cell_h))
    for col in range(cols + 1):
        page.draw_line((x0 + col * cell_w, y0), (x0 + col * cell_w, y0 + rows * cell_h))

    table_cells = [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]
    for row, cells in enumerate(table_cells):
        for col, value in enumerate(cells):
            page.insert_text((x0 + col * cell_w + 5, y0 + row * cell_h + 14), value, fontsize=9)

    page.insert_text((72, 260), outro_text)

    data: bytes = document.tobytes()
    document.close()
    return data


def make_docx_bytes(text: str) -> bytes:
    """Build a minimal in-memory OOXML `.docx` containing one paragraph."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
    ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>""",
        )
        archive.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
</w:document>""",
        )
    return buffer.getvalue()
