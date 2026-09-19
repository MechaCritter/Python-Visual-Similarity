"""Render the alert blocks of GitHub as admonitions.

The tutorial notebooks are read on GitHub as well as on the documentation site,
so their Markdown cells write admonitions in the syntax GitHub understands::

    > [!NOTE]
    >
    > Text of the note.

However, MyST knows no such syntax and renders the block as a quotation that
opens with the literal marker instead. Hence, this extension replaces every
quotation that opens with a marker by the admonition the marker names.
"""

from typing import Any

from docutils import nodes
from sphinx.application import Sphinx

# The admonition of every marker that GitHub defines.
ADMONITIONS: dict[str, type[nodes.Element]] = {
    "[!NOTE]": nodes.note,
    "[!TIP]": nodes.tip,
    "[!IMPORTANT]": nodes.important,
    "[!WARNING]": nodes.warning,
    "[!CAUTION]": nodes.caution,
}


def opening_marker(quotation: nodes.block_quote) -> str | None:
    """Read the alert marker that a quotation opens with.

    :param quotation: The quotation to read.
    :return: The marker, or ``None`` if the quotation opens without one.
    """
    if not quotation.children:
        return None
    first = quotation.children[0]
    if not isinstance(first, nodes.paragraph):
        return None
    text = first.astext()
    return next(
        (
            marker
            for marker in ADMONITIONS
            if text == marker or text.startswith(f"{marker}\n")
        ),
        None,
    )


def strip_marker(paragraph: nodes.paragraph, marker: str) -> bool:
    """Remove the marker from the start of a paragraph.

    :param paragraph: The paragraph that the marker opens.
    :param marker: The marker to remove.
    :return: Whether the paragraph holds nothing else.
    """
    opening = paragraph.children[0]
    remainder = opening.astext()[len(marker) :].lstrip("\n")
    if remainder:
        paragraph.replace(opening, nodes.Text(remainder))
        return False
    paragraph.remove(opening)
    return not paragraph.children


def as_admonition(quotation: nodes.block_quote, marker: str) -> nodes.Element:
    """Build the admonition that a quotation stands for.

    :param quotation: The quotation that opens with the marker.
    :param marker: The marker that the quotation opens with.
    :return: The admonition holding the content of the quotation.
    """
    children = list(quotation.children)
    opening = children[0]
    if isinstance(opening, nodes.paragraph) and strip_marker(opening, marker):
        children = children[1:]
    return ADMONITIONS[marker]("", *children)


def convert_alerts(app: Sphinx, doctree: nodes.document) -> None:
    """Replace every alert block of a document by its admonition.

    :param app: The Sphinx application.
    :param doctree: The document to convert.
    """
    for quotation in list(doctree.findall(nodes.block_quote)):
        marker = opening_marker(quotation)
        if marker is not None:
            quotation.parent.replace(quotation, as_admonition(quotation, marker))


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the conversion with Sphinx.

    :param app: The Sphinx application.
    :return: The extension metadata.
    """
    app.connect("doctree-read", convert_alerts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
