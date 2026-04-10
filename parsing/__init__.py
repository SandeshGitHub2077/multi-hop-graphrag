import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SECTION_ID_PATTERN = re.compile(r"§\s*(\d+(?:\.\w+)*)")
RFC_SECTION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(\w|\b)", re.MULTILINE)
SECTION_REF_PATTERN = re.compile(r"(?:see|refer to|as described in|according to|reference(?:d)? to|chapter|section)\s+§?\s*(\d+(?:\.\w+)*)", re.IGNORECASE)

RFC2119_KEYWORDS = re.compile(
    r"\b(MUST|SHOULD|MAY|REQUIRED|RECOMMENDED|OPTIONAL)"
    r"\s+(?:NOT\s+)?(?:SHALL|MUST|should|may)"
    r"(?:\s+be\s+|\s+not\s+be\s+)?",
    re.IGNORECASE
)

NOTE_REF_PATTERN = re.compile(
    r"(?:note|remark|example|note\s+that|see\s+note)\s+[:\-]?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE
)

RFC_REF_PATTERN = re.compile(
    r"(?:RFC|rfc)\s*(\d+)",
    re.IGNORECASE
)


@dataclass
class Section:
    doc_id: str
    section_id: str
    content: str
    references: list[str]


class DocumentParser:
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)

    def parse_file(self, file_path: str) -> list[Section]:
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return self._parse_pdf(path)
        elif ext == ".docx":
            return self._parse_docx(path)
        elif ext == ".txt":
            return self._parse_txt(path)
        elif ext in [".md", ".markdown"]:
            return self._parse_markdown(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _parse_pdf(self, path: Path) -> list[Section]:
        import fitz
        
        doc_id = path.stem
        sections = []
        
        doc = fitz.open(path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        doc.close()
        
        raw_text = "\n".join(full_text)
        sections = self._extract_sections(raw_text, doc_id)
        
        return sections

    def _parse_docx(self, path: Path) -> list[Section]:
        from docx import Document
        
        doc_id = path.stem
        doc = Document(path)
        
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        raw_text = "\n".join(full_text)
        sections = self._extract_sections(raw_text, doc_id)
        
        return sections

    def _parse_txt(self, path: Path) -> list[Section]:
        doc_id = path.stem
        raw_text = path.read_text(encoding="utf-8")
        return self._extract_sections(raw_text, doc_id)

    def _parse_markdown(self, path: Path) -> list[Section]:
        doc_id = path.stem
        raw_text = path.read_text(encoding="utf-8")
        return self._extract_sections(raw_text, doc_id)

    def _extract_sections(self, raw_text: str, doc_id: str) -> list[Section]:
        if doc_id.lower().startswith("rfc"):
            return self._extract_rfc_sections(raw_text, doc_id)
        return self._extract_cfr_sections(raw_text, doc_id)
    
    def _extract_rfc_sections(self, raw_text: str, doc_id: str) -> list[Section]:
        lines = raw_text.split("\n")
        sections = []
        current_section_id: Optional[str] = None
        current_content_lines = []
        
        for line in lines:
            section_match = RFC_SECTION_PATTERN.match(line)
            
            if section_match:
                if current_section_id and current_content_lines:
                    content = "\n".join(current_content_lines).strip()
                    if content:
                        sections.append(Section(
                            doc_id=doc_id,
                            section_id=current_section_id,
                            content=content,
                            references=self._extract_references(content)
                        ))
                
                potential_id = section_match.group(1)
                if self._is_valid_section_id(potential_id):
                    current_section_id = potential_id
                    current_content_lines = [line]
                else:
                    current_content_lines.append(line)
            else:
                if current_section_id:
                    current_content_lines.append(line)
                elif line.strip():
                    if sections and sections[-1].section_id == current_section_id:
                        sections[-1].content += "\n" + line
                    elif not sections:
                        pass
        
        if current_section_id and current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if content:
                sections.append(Section(
                    doc_id=doc_id,
                    section_id=current_section_id,
                    content=content,
                    references=self._extract_references(content)
                ))
        
        return sections
    
    def _extract_cfr_sections(self, raw_text: str, doc_id: str) -> list[Section]:
        lines = raw_text.split("\n")
        sections = []
        current_section_id: Optional[str] = None
        current_content_lines = []
        
        for line in lines:
            section_match = SECTION_ID_PATTERN.search(line)
            
            if section_match:
                if current_section_id and current_content_lines:
                    content = "\n".join(current_content_lines).strip()
                    if content:
                        sections.append(Section(
                            doc_id=doc_id,
                            section_id=current_section_id,
                            content=content,
                            references=self._extract_references(content)
                        ))
                
                potential_id = section_match.group(1)
                if self._is_valid_section_id(potential_id):
                    current_section_id = potential_id
                    current_content_lines = [line]
                else:
                    current_content_lines.append(line)
            else:
                if current_section_id:
                    current_content_lines.append(line)
                elif line.strip():
                    if sections and sections[-1].section_id == current_section_id:
                        sections[-1].content += "\n" + line
                    elif not sections:
                        pass
        
        if current_section_id and current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if content:
                sections.append(Section(
                    doc_id=doc_id,
                    section_id=current_section_id,
                    content=content,
                    references=self._extract_references(content)
                ))
        
        return sections

    def _is_valid_section_id(self, text: str) -> bool:
        if not text:
            return False
        if not text[0].isdigit():
            return False
        parts = text.split('.')
        if not parts[0].isdigit():
            return False
        return True

    def _extract_references(self, text: str) -> list[str]:
        references = set()

        matches = SECTION_REF_PATTERN.findall(text)
        for match in matches:
            if self._is_valid_section_id(match) and match not in references:
                references.add(match)

        note_matches = NOTE_REF_PATTERN.findall(text)
        for match in note_matches:
            if self._is_valid_section_id(match) and match not in references:
                references.add(f"note:{match}")

        return list(references)

    def parse_all(self) -> list[Section]:
        all_sections = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                try:
                    sections = self.parse_file(str(file_path))
                    all_sections.extend(sections)
                    print(f"Parsed {len(sections)} sections from {file_path.name}")
                except Exception as e:
                    print(f"Error parsing {file_path.name}: {e}")
        
        return all_sections
