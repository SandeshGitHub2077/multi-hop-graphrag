import os
from typing import Optional
from neo4j import GraphDatabase


class Neo4jGraph:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

    def connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section)
                REQUIRE s.section_id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document)
                REQUIRE d.doc_id IS UNIQUE
            """)

    def clear_all(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_section(self, section_id: str, content: str, doc_id: str, metadata: dict = None):
        import json
        ref_list = metadata.get("references", []) if metadata else []
        refs_json = json.dumps(ref_list) if ref_list else "[]"
        
        with self.driver.session() as session:
            session.run("""
                MERGE (s:Section {section_id: $section_id})
                SET s.content = $content,
                    s.doc_id = $doc_id,
                    s.references = $refs
            """, section_id=section_id, content=content, doc_id=doc_id, refs=refs_json)

    def upsert_document(self, doc_id: str, doc_type: str = None, metadata: dict = None):
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.doc_type = $doc_type
            """, doc_id=doc_id, doc_type=doc_type)

    def create_reference_relationship(
        self, from_section_id: str, to_section_id: str, ref_type: str = "REFERENCES"
    ):
        with self.driver.session() as session:
            session.run("""
                MATCH (from:Section {section_id: $from_id})
                MATCH (to:Section {section_id: $to_id})
                MERGE (from)-[r:REFERENCES]->(to)
                SET r.ref_type = $ref_type
            """, from_id=from_section_id, to_id=to_section_id, ref_type=ref_type)

    def create_doc_relationship(self, section_id: str, doc_id: str):
        with self.driver.session() as session:
            session.run("""
                MATCH (s:Section {section_id: $section_id})
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (s)-[r:PART_OF]->(d)
            """, section_id=section_id, doc_id=doc_id)

    def get_section(self, section_id: str) -> Optional[dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})
                RETURN s.section_id as section_id, s.content as content, s.doc_id as doc_id
            """, section_id=section_id)
            record = result.single()
            if record:
                return dict(record)
            return None

    def get_references(self, section_id: str, direction: str = "out") -> list[dict]:
        with self.driver.session() as session:
            if direction == "out":
                result = session.run("""
                    MATCH (s:Section {section_id: $section_id})-[r:REFERENCES]->(ref:Section)
                    RETURN ref.section_id as section_id, ref.content as content
                """, section_id=section_id)
            else:
                result = session.run("""
                    MATCH (s:Section {section_id: $section_id})<-[r:REFERENCES]-(ref:Section)
                    RETURN ref.section_id as section_id, ref.content as content
                """, section_id=section_id)
            
            return [dict(record) for record in result]

    def get_reference_density(self, section_id: str) -> int:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})
                OPTIONAL MATCH (s)-[r1:REFERENCES]->(:Section)
                OPTIONAL MATCH (:Section)-[r2:REFERENCES]->(s)
                RETURN count(r1) + count(r2) as density
            """, section_id=section_id)
            record = result.single()
            return record["density"] if record else 0

    def get_all_sections(self) -> list[dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section)
                RETURN s.section_id as section_id, s.content as content, s.doc_id as doc_id
            """)
            return [dict(record) for record in result]

    def get_neighbors(self, section_id: str, doc_id: str = None, depth: int = 1) -> list[dict]:
        with self.driver.session() as session:
            if doc_id:
                result = session.run("""
                    MATCH (s:Section {section_id: $section_id, doc_id: $doc_id})
                    MATCH path = (s)-[:REFERENCES*1..""" + str(depth) + """]->(neighbor:Section)
                    RETURN neighbor.section_id as section_id, neighbor.content as content,
                           neighbor.doc_id as doc_id, length(path) as distance
                """, section_id=section_id, doc_id=doc_id)
            else:
                result = session.run("""
                    MATCH (s:Section {section_id: $section_id})
                    MATCH path = (s)-[:REFERENCES*1..""" + str(depth) + """]->(neighbor:Section)
                    RETURN neighbor.section_id as section_id, neighbor.content as content,
                           neighbor.doc_id as doc_id, length(path) as distance
                """, section_id=section_id)
            return [dict(record) for record in result]

    def get_all_documents(self) -> list[str]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.doc_id as doc_id
            """)
            return [record["doc_id"] for record in result]

    def get_section_ids(self) -> set[str]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section)
                RETURN s.section_id as section_id
            """)
            return {record["section_id"] for record in result}

    def section_exists(self, section_id: str) -> bool:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})
                RETURN s.section_id
            """, section_id=section_id)
            return result.single() is not None

    def get_section_content(self, section_id: str) -> Optional[str]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})
                RETURN s.content as content
            """, section_id=section_id)
            record = result.single()
            return record["content"] if record else None
