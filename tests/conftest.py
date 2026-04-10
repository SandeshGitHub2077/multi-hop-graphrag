import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_pdf_path():
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_docx_path():
    return Path(__file__).parent / "fixtures" / "sample.docx"


@pytest.fixture
def sample_txt_path():
    return Path(__file__).parent / "fixtures" / "sample.txt"


@pytest.fixture
def sample_md_path():
    return Path(__file__).parent / "fixtures" / "sample.md"


@pytest.fixture
def cfr_section_text():
    return """§ 1.1 Purpose.
The purpose of this part is to establish regulations for food safety.

§ 1.2 Scope.
These regulations apply to all food facilities.

§ 1.3 References.
See § 1.1 for purpose. Refer to § 1.2 for scope."""


@pytest.fixture
def parsed_sections():
    from parsing import Section
    return [
        Section(
            doc_id="test_doc",
            section_id="1.1",
            content="The purpose of this part is to establish regulations for food safety.",
            references=["1.2"]
        ),
        Section(
            doc_id="test_doc",
            section_id="1.2",
            content="These regulations apply to all food facilities.",
            references=[]
        ),
    ]
