import logging
import os
from typing import List
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
try:
    driver = GraphDatabase.driver(NEO4J_URI or "", auth=(NEO4J_USER or "", NEO4J_PASSWORD or ""))
    driver.verify_connectivity()
    logger.info("Neo4j connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    driver = None


class KGClient:

    def __init__(self, driver):
        self.driver = driver

    # --------------------------------------
    # ADD DOCUMENT
    # --------------------------------------
    def add_document(self, source_id: str, metadata: dict) -> bool:
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return False

        try:
            with self.driver.session() as session:
                session.execute_write(self._create_doc, source_id, metadata)
            logger.info(f"Document added to KG: {source_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {source_id} to KG: {e}")
            return False

    @staticmethod
    def _create_doc(tx, source_id: str, metadata: dict):
        """
        Store document metadata in Neo4j as a node:
            (:Document {id: "...", ...metadata })
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d += $md,
            d.updated_at = timestamp()
        """
        tx.run(query, id=source_id, md=metadata)


    def add_entity(self, name: str, ent_type: str = 'Entity') -> bool:
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return False

        try:
            with self.driver.session() as session:
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.updated_at = timestamp()
                        """,
                        name=name,
                        type=ent_type
                    )
                )
            logger.info(f"Entity added to KG: {name} ({ent_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to add entity {name}: {e}")
            return False


    def link_mention(self, source_id: str, entity_name: str) -> bool:
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return False

        try:
            with self.driver.session() as session:
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        MATCH (d:Document {id: $doc})
                        MATCH (e:Entity {name: $ename})
                        MERGE (d)-[:MENTIONS]->(e)
                        """,
                        doc=source_id,
                        ename=entity_name
                    )
                )
            logger.info(f"Linked document {source_id} to entity {entity_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to link: {e}")
            return False

    # --------------------------------------
    # FIND ENTITIES BY SEARCH
    # --------------------------------------
    def find_entities_by_text(self, text: str, limit: int = 10) -> List[str]:
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($q)
                    RETURN e.name AS name
                    ORDER BY size(e.name)
                    LIMIT $limit
                    """,
                    q=text,
                    limit=limit
                )
                return [record["name"] for record in result]
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []

    # --------------------------------------
    # FIND RELATED DOCUMENTS (NEIGHBORS)
    # --------------------------------------
    def get_neighbor_documents(self, entity_name: str, hops: int = 1, limit: int = 50) -> List[Document]:
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return []

        # enforce valid hop range
        hops = max(1, min(hops, 5))

        try:
            query = f"""
            MATCH (e:Entity {{name: $name}})
            MATCH p=(e)-[*1..{hops}]-(d:Document)
            RETURN DISTINCT d
            LIMIT $limit
            """

            with self.driver.session() as session:
                result = session.run(query, name=entity_name, limit=limit)

                docs = []
                for record in result:
                    props = dict(record["d"])

                    # Determine Page Content
                    content = (
                        props.get("text")
                        or props.get("ocr_text")
                        or props.get("caption")
                        or props.get("transcription")
                        or f"Document {props.get('id')}"
                    )

                    docs.append(
                        Document(
                            page_content=content,
                            metadata=props
                        )
                    )

                return docs

        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


kg_client = KGClient(driver) if driver else None

if kg_client is None:
    logger.warning("KG client is not available - Neo4j operations will be skipped")