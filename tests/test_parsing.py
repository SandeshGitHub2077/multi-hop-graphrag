import pytest
from parsing import DocumentParser, Section


class TestDocumentParser:
    def test_parser_initialization(self):
        parser = DocumentParser("Data")
        assert parser.data_dir.name == "Data"

    def test_parse_pdf(self, sample_pdf_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_pdf_path))
        
        assert len(sections) > 0
        assert all(isinstance(s, Section) for s in sections)
        
        section_ids = [s.section_id for s in sections]
        assert "1.1" in section_ids or "1.2" in section_ids

    def test_parse_txt(self, sample_txt_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_txt_path))
        
        assert len(sections) > 0
        assert all(isinstance(s, Section) for s in sections)

    def test_parse_markdown(self, sample_md_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_md_path))
        
        assert len(sections) > 0
        assert all(isinstance(s, Section) for s in sections)

    def test_parse_unsupported_file(self, tmp_path):
        parser = DocumentParser()
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse_file(str(test_file))

    def test_section_extraction_with_cfr_format(self, sample_pdf_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_pdf_path))
        
        section_ids = [s.section_id for s in sections]
        assert any("1.1" in sid for sid in section_ids), "Should extract CFR section 1.1"

    def test_reference_extraction(self, sample_pdf_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_pdf_path))
        
        refs_found = False
        for s in sections:
            if s.references:
                refs_found = True
                break
        
        assert refs_found, "Should extract references like § 1.1, § 1.2"


class TestSectionDetection:
    def test_section_id_pattern(self):
        parser = DocumentParser()
        
        assert parser._is_valid_section_id("1.1") == True
        assert parser._is_valid_section_id("405.2414") == True
        assert parser._is_valid_section_id("1.100.1") == True
        assert parser._is_valid_section_id("abc") == False
        assert parser._is_valid_section_id("") == False

    def test_section_content_not_empty(self, sample_pdf_path):
        parser = DocumentParser()
        sections = parser.parse_file(str(sample_pdf_path))
        
        for s in sections:
            assert s.content.strip(), f"Section {s.section_id} should have content"
            assert s.doc_id, "Section should have doc_id"


class TestReferenceExtraction:
    def test_extract_references_from_text(self):
        parser = DocumentParser()
        
        text = "See § 1.1 for purpose. Refer to § 1.2 for scope."
        refs = parser._extract_references(text)
        
        assert "1.1" in refs or "1.2" in refs
