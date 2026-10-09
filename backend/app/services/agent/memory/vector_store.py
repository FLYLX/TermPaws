import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Literal

import chromadb
from chromadb import Collection

from app.core.config import settings

logger = logging.getLogger(__name__)

MemoryType = Literal["fact", "preference", "error", "context"]

MEMORY_TYPES = {
    "fact": "事实信息：用户名、路径、配置值等",
    "preference": "用户偏好：代码风格、工具选择等",
    "error": "错误记录：已知问题和解决方案",
    "context": "上下文：项目结构、依赖关系等",
}

DEFAULT_MEMORY_TTL_DAYS = 30
DEFAULT_MEMORY_TTL_DAYS_BY_TYPE = {
    "fact": 90,
    "context": 30,
}
IMMORTAL_MEMORY_TYPES = {"preference", "error"}
EXPIRED_MEMORY_COMPACTION_THRESHOLD = 10
EXPIRED_MEMORY_SUMMARY_MAX_CHARS = 800
DEDUP_THRESHOLD = 0.95
SUMMARIZE_THRESHOLD = 10
MEMORY_SUMMARY_PREFIX_STRIP_RE = re.compile(r"^(?:历史\S{0,6}记忆摘要[:：]\s*)+")
MEMORY_TRANSPORT_MARKUP_RE = re.compile(
    r"\[CQ:[^\]]*\]|\[Robot message;[^\]]*\]|\[(?:Current QQ message|Pending QQ messages)[^\]]*\]",
    re.IGNORECASE,
)
DEFAULT_RECALL_CANDIDATE_MULTIPLIER = 4
FALLBACK_EMBEDDING_DIMENSION = 384
MEMORY_COLLECTION_NAME = "handler_memories"
EMBEDDING_MODEL_METADATA_KEY = "embedding_model"
EMBEDDING_DIMENSION_METADATA_KEY = "embedding_dimension"
EMBEDDING_REINDEX_BATCH_SIZE = 128
# 单次 onnx 推理的批大小上限：transformer 中间激活内存 ≈ batch×seq×hidden×layers，
# 批 219 时峰值 ~2.7GB，且 onnxruntime arena 抓到后永不释放（RSS 永久高水位）。
EMBEDDING_ENCODE_BATCH_SIZE = 8
LEXICAL_INDEX_FILE_NAME = "memory_fts.sqlite3"
HYBRID_LEXICAL_WEIGHT = 0.18


class EmbeddingService:
    _instance: "EmbeddingService | None" = None
    _model: Any | None = None
    _model_name: str | None = None
    _load_error: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _ensure_model(self):
        model_name = settings.EMBEDDING_MODEL_NAME
        if self._model_name != model_name:
            self._model = None
            self._model_name = model_name
            self._load_error = None

        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        if self._model is None:
            logger.info("[Embedding] Loading local model %s...", model_name)
            from fastembed import TextEmbedding

            try:
                # 本地缓存命中则直接用，缺失则自动下载（默认国内镜像源）
                self._model = TextEmbedding(model_name=model_name)
                logger.info("[Embedding] Local model loaded")
            except Exception as exc:
                self._load_error = str(exc)
                logger.warning(
                    "[Embedding] 本地模型不可用且下载失败，本次使用降级检索: %s",
                    exc,
                )
                raise RuntimeError(self._load_error) from exc

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        self._ensure_model()
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), EMBEDDING_ENCODE_BATCH_SIZE):
            batch = texts[offset : offset + EMBEDDING_ENCODE_BATCH_SIZE]
            for embedding in self._model.embed(batch):
                vector = [float(v) for v in embedding]
                norm = math.sqrt(sum(v * v for v in vector)) or 1.0
                vectors.append([v / norm for v in vector])
        return vectors

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def dimension(self) -> int:
        self._ensure_model()
        get_dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(get_dimension):
            return int(get_dimension())
        return len(self.encode_single("dimension probe"))


class VectorStoreService:
    _instance: "VectorStoreService | None" = None
    _client: Any = None
    _collection: Collection | None = None
    _embedding_service: EmbeddingService | None = None
    _collection_embedding_model: str | None = None
    _maintenance_lock = threading.Lock()
    _lexical_index_lock = threading.RLock()
    _lexical_index_path: str | None = None
    _lexical_index_available = True
    _maintenance_items: set[str] = set()
    _maintenance_last_checked: dict[str, float] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _ensure_initialized(self):
        if self._client is not None:
            return

        persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)

        try:
            logger.info(
                f"[VectorStore] Initializing ChromaDB with persistence at {persist_dir}"
            )
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._embedding_service = EmbeddingService()
            self._initialize_memory_collection()
            self._sync_lexical_index()
        except Exception as exc:
            self._client = object()
            self._collection = None
            self._collection_embedding_model = None
            logger.warning(
                "[VectorStore] ChromaDB unavailable; using JSON fallback memory store: %s",
                exc,
            )
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        if self._collection is not None:
            logger.info("[VectorStore] ChromaDB initialized with persistence")

    @staticmethod
    def _collection_name_for_model(model_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
        return f"{MEMORY_COLLECTION_NAME}_{slug[:34]}_{digest}"

    @staticmethod
    def _collection_metadata_for_model(
        model_name: str,
        *,
        dimension: int | None = None,
        migration_complete: bool,
        include_index_config: bool = False,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "description": "Long-term memory for items",
            EMBEDDING_MODEL_METADATA_KEY: model_name,
            "migration_complete": migration_complete,
        }
        if include_index_config:
            metadata["hnsw:space"] = "cosine"
        if dimension is not None:
            metadata[EMBEDDING_DIMENSION_METADATA_KEY] = dimension
        return metadata

    @staticmethod
    def _collection_count(collection: Any) -> int:
        try:
            return int(collection.count())
        except Exception:
            return 0

    def _find_memory_collection_source(self, active_name: str) -> Any | None:
        if not hasattr(self._client, "list_collections"):
            return None

        candidates: list[Any] = []
        try:
            collections = self._client.list_collections()
        except Exception as exc:
            logger.warning("[VectorStore] Failed to list memory collections: %s", exc)
            return None

        for collection in collections:
            name = str(getattr(collection, "name", "") or "")
            metadata = dict(getattr(collection, "metadata", {}) or {})
            if not name.startswith(MEMORY_COLLECTION_NAME) or name == active_name:
                continue
            if metadata.get("superseded_by"):
                continue
            if self._collection_count(collection) <= 0:
                continue
            candidates.append(collection)

        if not candidates:
            return None
        return max(candidates, key=self._collection_count)

    def _cleanup_superseded_memory_collections(self, active_name: str) -> None:
        try:
            collections = self._client.list_collections()
        except Exception:
            return
        for collection in collections:
            name = str(getattr(collection, "name", "") or "")
            metadata = dict(getattr(collection, "metadata", {}) or {})
            if name == active_name or metadata.get("superseded_by") != active_name:
                continue
            try:
                self._client.delete_collection(name)
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to remove superseded memory index %s: %s",
                    name,
                    exc,
                )

    def _initialize_memory_collection(self) -> None:
        model_name = settings.EMBEDDING_MODEL_NAME
        active_name = self._collection_name_for_model(model_name)
        supports_migration = all(
            hasattr(self._client, method)
            for method in (
                "list_collections",
                "delete_collection",
                "get_or_create_collection",
            )
        )
        if not supports_migration:
            self._collection = self._client.get_or_create_collection(
                name=MEMORY_COLLECTION_NAME,
                metadata={"description": "Long-term memory for items"},
            )
            self._collection_embedding_model = model_name
            return

        active = self._client.get_or_create_collection(
            name=active_name,
            metadata=self._collection_metadata_for_model(
                model_name,
                migration_complete=False,
                include_index_config=True,
            ),
        )
        active_metadata = dict(getattr(active, "metadata", {}) or {})
        source = self._find_memory_collection_source(active_name)
        active_count = self._collection_count(active)
        migration_complete = active_metadata.get("migration_complete") is True

        if active_count > 0 and not migration_complete and source is not None:
            logger.warning(
                "[VectorStore] Removing incomplete embedding index %s before retrying migration",
                active_name,
            )
            self._client.delete_collection(active_name)
            active = self._client.get_or_create_collection(
                name=active_name,
                metadata=self._collection_metadata_for_model(
                    model_name,
                    migration_complete=False,
                    include_index_config=True,
                ),
            )
            active_count = 0

        if active_count == 0 and source is not None:
            try:
                self._migrate_memory_collection(source, active, model_name=model_name)
                self._collection = active
                self._collection_embedding_model = model_name
                return
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to rebuild memory embeddings with %s; using the previous index with lexical recall: %s",
                    model_name,
                    exc,
                )
                self._client.delete_collection(active_name)
                self._collection = source
                self._collection_embedding_model = str(
                    (getattr(source, "metadata", {}) or {}).get(
                        EMBEDDING_MODEL_METADATA_KEY,
                        "legacy-incompatible-index",
                    )
                )
                return

        if active_count == 0 and not migration_complete:
            dimension = self._embedding_service.dimension()
            active.modify(
                metadata=self._collection_metadata_for_model(
                    model_name,
                    dimension=dimension,
                    migration_complete=True,
                )
            )

        self._cleanup_superseded_memory_collections(active_name)
        self._collection = active
        self._collection_embedding_model = model_name

    def _migrate_memory_collection(
        self,
        source: Any,
        target: Any,
        *,
        model_name: str,
    ) -> None:
        records = source.get(include=["documents", "metadatas"])
        ids = list(records.get("ids") or [])
        documents = [str(document or "") for document in records.get("documents") or []]
        metadatas = list(records.get("metadatas") or [])
        if not ids:
            return
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise RuntimeError("memory collection returned inconsistent migration data")

        embeddings = self._embedding_service.encode(documents)
        if len(embeddings) != len(ids):
            raise RuntimeError("embedding model returned an incomplete migration batch")
        dimension = (
            len(embeddings[0]) if embeddings else self._embedding_service.dimension()
        )

        for offset in range(0, len(ids), EMBEDDING_REINDEX_BATCH_SIZE):
            end = offset + EMBEDDING_REINDEX_BATCH_SIZE
            target.add(
                ids=ids[offset:end],
                embeddings=embeddings[offset:end],
                documents=documents[offset:end],
                metadatas=metadatas[offset:end],
            )

        migrated_count = self._collection_count(target)
        if migrated_count != len(ids):
            raise RuntimeError(
                f"memory migration count mismatch: expected {len(ids)}, got {migrated_count}"
            )

        target.modify(
            metadata=self._collection_metadata_for_model(
                model_name,
                dimension=dimension,
                migration_complete=True,
            )
        )
        source_name = str(getattr(source, "name", "") or "")
        try:
            self._client.delete_collection(source_name)
        except Exception as exc:
            source_metadata = dict(getattr(source, "metadata", {}) or {})
            source_metadata.pop("hnsw:space", None)
            source_metadata["superseded_by"] = str(getattr(target, "name", "") or "")
            source.modify(metadata=source_metadata)
            logger.warning(
                "[VectorStore] Kept superseded memory index %s because cleanup failed: %s",
                source_name,
                exc,
            )
        logger.info(
            "[VectorStore] Rebuilt %s memory embeddings from %s to %s with %s",
            migrated_count,
            source_name or "legacy",
            getattr(target, "name", "active"),
            model_name,
        )

    def _collection_supports_current_model(self) -> bool:
        return self._collection_embedding_model in {
            None,
            settings.EMBEDDING_MODEL_NAME,
        }

    def _lexical_index_file(self) -> str:
        return os.path.join(settings.CHROMA_PERSIST_DIR, LEXICAL_INDEX_FILE_NAME)

    def _ensure_lexical_index(self) -> bool:
        path = self._lexical_index_file()
        with self._lexical_index_lock:
            if self._lexical_index_path == path:
                return self._lexical_index_available
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with sqlite3.connect(path, timeout=10) as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                    columns = {
                        row[1]
                        for row in connection.execute(
                            "PRAGMA table_info(memory_fts)"
                        ).fetchall()
                    }
                    if columns and "handler_id" not in columns:
                        connection.execute("DROP TABLE memory_fts")
                    connection.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                            memory_id UNINDEXED,
                            handler_id UNINDEXED,
                            memory_type UNINDEXED,
                            content UNINDEXED,
                            metadata_json UNINDEXED,
                            token_text,
                            tokenize='unicode61'
                        )
                        """
                    )
                self._lexical_index_available = True
            except Exception as exc:
                self._lexical_index_available = False
                logger.warning("[VectorStore] SQLite FTS5 unavailable: %s", exc)
            self._lexical_index_path = path
            return self._lexical_index_available

    @classmethod
    def _lexical_token_text(cls, text: str) -> str:
        return " ".join(sorted(cls._lexical_tokens(text)))

    def _upsert_lexical_memory(
        self,
        *,
        memory_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        if not self._ensure_lexical_index():
            return
        path = self._lexical_index_file()
        with self._lexical_index_lock:
            try:
                with sqlite3.connect(path, timeout=10) as connection:
                    connection.execute(
                        "DELETE FROM memory_fts WHERE memory_id = ?",
                        (str(memory_id),),
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_fts(
                            memory_id,
                            handler_id,
                            memory_type,
                            content,
                            metadata_json,
                            token_text
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(memory_id),
                            str(metadata.get("handler_id") or ""),
                            str(metadata.get("memory_type") or "fact"),
                            str(content or ""),
                            json.dumps(metadata, ensure_ascii=False, default=str),
                            self._lexical_token_text(content),
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to update lexical memory %s: %s",
                    memory_id,
                    exc,
                )

    def _delete_lexical_memories(
        self,
        *,
        memory_id: str | None = None,
        handler_id: str | None = None,
    ) -> None:
        if not self._ensure_lexical_index():
            return
        if memory_id is None and handler_id is None:
            return
        field = "memory_id" if memory_id is not None else "handler_id"
        value = str(memory_id if memory_id is not None else handler_id)
        with self._lexical_index_lock:
            try:
                with sqlite3.connect(
                    self._lexical_index_file(), timeout=10
                ) as connection:
                    connection.execute(
                        f"DELETE FROM memory_fts WHERE {field} = ?",
                        (value,),
                    )
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to delete lexical memories by %s: %s",
                    field,
                    exc,
                )

    def _sync_lexical_index(self) -> None:
        if not self._ensure_lexical_index():
            return
        records: list[dict[str, Any]] = []
        if self._collection is not None:
            try:
                result = self._collection.get(include=["documents", "metadatas"])
                for index, memory_id in enumerate(result.get("ids") or []):
                    records.append(
                        {
                            "id": str(memory_id),
                            "content": str(
                                (result.get("documents") or [])[index] or ""
                            ),
                            "metadata": (result.get("metadatas") or [])[index] or {},
                        }
                    )
            except Exception as exc:
                logger.warning("[VectorStore] Failed to sync Chroma into FTS5: %s", exc)
        records.extend(self._load_fallback_memories())

        with self._lexical_index_lock:
            try:
                with sqlite3.connect(
                    self._lexical_index_file(), timeout=10
                ) as connection:
                    connection.execute("DELETE FROM memory_fts")
                    for record in records:
                        metadata = record.get("metadata") or {}
                        connection.execute(
                            """
                            INSERT INTO memory_fts(
                                memory_id,
                                handler_id,
                                memory_type,
                                content,
                                metadata_json,
                                token_text
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(record.get("id") or ""),
                                str(metadata.get("handler_id") or ""),
                                str(metadata.get("memory_type") or "fact"),
                                str(record.get("content") or ""),
                                json.dumps(metadata, ensure_ascii=False, default=str),
                                self._lexical_token_text(
                                    str(record.get("content") or "")
                                ),
                            ),
                        )
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to rebuild SQLite FTS5 index: %s", exc
                )

    def _search_lexical_index(
        self,
        *,
        handler_id: str,
        query: str,
        memory_type: MemoryType | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self._ensure_lexical_index() or limit <= 0:
            return []
        tokens = sorted(
            self._lexical_tokens(query),
            key=lambda token: (len(token), token),
            reverse=True,
        )[:16]
        if not tokens:
            return []
        match_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        sql = (
            "SELECT memory_id, content, metadata_json "
            "FROM memory_fts WHERE memory_fts MATCH ? AND handler_id = ?"
        )
        params: list[Any] = [match_query, str(handler_id)]
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(str(memory_type))
        sql += " ORDER BY bm25(memory_fts) LIMIT ?"
        params.append(int(limit))

        try:
            with sqlite3.connect(self._lexical_index_file(), timeout=10) as connection:
                rows = connection.execute(sql, params).fetchall()
        except Exception as exc:
            logger.warning("[VectorStore] Failed to search SQLite FTS5: %s", exc)
            return []

        memories: list[dict[str, Any]] = []
        for memory_id, content, metadata_json in rows:
            try:
                metadata = json.loads(str(metadata_json or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            lexical_similarity = self._lexical_similarity(query, str(content or ""))
            memories.append(
                {
                    "id": str(memory_id or ""),
                    "content": str(content or ""),
                    "metadata": self._visible_memory_metadata(metadata),
                    "distance": 1.0 - lexical_similarity,
                    "lexical_similarity": lexical_similarity,
                }
            )
        return memories

    def _try_encode_single(self, text: str) -> list[float] | None:
        try:
            return self._embedding_service.encode_single(text)
        except Exception as exc:
            logger.warning("[VectorStore] Embedding unavailable: %s", exc)
            return None

    def _encode_single_with_fallback(self, text: str) -> tuple[list[float], bool]:
        try:
            return self._embedding_service.encode_single(text), False
        except Exception as e:
            logger.warning("[VectorStore] Using fallback memory embedding: %s", e)
            return self._fallback_embedding(text), True

    def _try_encode(self, texts: list[str]) -> list[list[float]] | None:
        try:
            return self._embedding_service.encode(texts)
        except Exception as e:
            logger.warning("[VectorStore] Embeddings unavailable: %s", e)
            return None

    @staticmethod
    def _fallback_embedding(text: str) -> list[float]:
        vector = [0.0] * FALLBACK_EMBEDDING_DIMENSION
        raw = str(text or "").encode("utf-8", errors="ignore")
        if not raw:
            return vector

        for index, byte in enumerate(raw):
            slot = (index * 131 + byte) % FALLBACK_EMBEDDING_DIMENSION
            sign = 1.0 if ((index + byte) % 2 == 0) else -1.0
            vector[slot] += sign * (1.0 + byte / 255.0)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def _fallback_store_path(self) -> str:
        return os.path.join(settings.CHROMA_PERSIST_DIR, "fallback_memories.json")

    def _load_fallback_memories(self) -> list[dict[str, Any]]:
        path = self._fallback_store_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            logger.warning("[VectorStore] Failed to load fallback memories: %s", exc)
            return []
        if not isinstance(data, list):
            return []
        return [entry for entry in data if isinstance(entry, dict)]

    def _write_fallback_memories(self, memories: list[dict[str, Any]]) -> None:
        path = self._fallback_store_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(memories, file, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_path, path)

    def _add_fallback_memory(
        self,
        *,
        memory_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        self._upsert_fallback_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
        )
        logger.info("[VectorStore] Added fallback memory %s", memory_id)
        return memory_id

    def _upsert_fallback_memory(
        self,
        *,
        memory_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> bool:
        memories = self._load_fallback_memories()
        record = {"id": memory_id, "content": content, "metadata": metadata}
        replaced = False
        for index, memory in enumerate(memories):
            if str(memory.get("id") or "") == str(memory_id):
                memories[index] = record
                replaced = True
                break
        if not replaced:
            memories.append(record)
        self._write_fallback_memories(memories)
        self._upsert_lexical_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
        )
        return True

    def _update_fallback_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        memories = self._load_fallback_memories()
        for memory in memories:
            if str(memory.get("id") or "") != str(memory_id):
                continue
            current_metadata = (
                memory.get("metadata")
                if isinstance(memory.get("metadata"), dict)
                else {}
            )
            if content is not None:
                memory["content"] = content
            if metadata:
                memory["metadata"] = {**current_metadata, **metadata}
            next_metadata = memory.get("metadata") or current_metadata
            if str(next_metadata.get("memory_type") or "") in IMMORTAL_MEMORY_TYPES:
                next_metadata.pop("expires_at", None)
                memory["metadata"] = next_metadata
            self._write_fallback_memories(memories)
            return True
        return False

    def _fallback_memories_for_handler(
        self,
        handler_id: str,
        memory_type: MemoryType | None = None,
    ) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        for memory in self._load_fallback_memories():
            metadata = (
                memory.get("metadata")
                if isinstance(memory.get("metadata"), dict)
                else {}
            )
            if str(metadata.get("handler_id") or "") != str(handler_id):
                continue
            if memory_type and metadata.get("memory_type") != memory_type:
                continue
            visible_metadata = self._visible_memory_metadata(metadata)
            memories.append(
                {
                    "id": str(memory.get("id") or ""),
                    "content": str(memory.get("content") or ""),
                    "metadata": visible_metadata,
                }
            )
        return [memory for memory in memories if memory["id"]]

    @staticmethod
    def _visible_memory_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        visible = dict(metadata)
        if str(visible.get("memory_type") or "") in IMMORTAL_MEMORY_TYPES:
            visible.pop("expires_at", None)
        return visible

    @staticmethod
    def _build_where_filter(
        handler_id: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> dict[str, Any] | None:
        filters: list[dict[str, Any]] = []
        if handler_id:
            filters.append({"handler_id": handler_id})
        if memory_type:
            filters.append({"memory_type": memory_type})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def _is_expired_memory(cls, metadata: dict[str, Any]) -> bool:
        if str(metadata.get("memory_type") or "") in IMMORTAL_MEMORY_TYPES:
            return False
        expires_at = cls._parse_datetime(metadata.get("expires_at"))
        if expires_at is None:
            return False
        now = datetime.now(expires_at.tzinfo) if expires_at.tzinfo else datetime.now()
        return expires_at < now

    @staticmethod
    def _resolved_memory_status(
        memory_type: str, content: str, metadata: dict[str, Any]
    ) -> str:
        status = str(metadata.get("status") or "").lower()
        if status:
            return status
        normalized = (content or "").strip()
        if memory_type == "task":
            if normalized.endswith("\uff08\u5df2\u5b8c\u6210\uff09"):
                return "completed"
            return "active"
        if memory_type == "error":
            if normalized.endswith("\uff08\u5df2\u89e3\u51b3\uff09"):
                return "resolved"
            return "active"
        return ""

    @classmethod
    def _is_inactive_status_memory(cls, content: str, metadata: dict[str, Any]) -> bool:
        memory_type = str(metadata.get("memory_type") or "")
        status = cls._resolved_memory_status(memory_type, content, metadata)
        return memory_type == "task" and status == "completed"

    @staticmethod
    def _normalize_memory_lifetime(
        metadata: dict[str, Any],
        *,
        memory_type: str,
        now: datetime,
        ttl_days: int | None,
    ) -> dict[str, Any]:
        normalized = dict(metadata)
        if memory_type in IMMORTAL_MEMORY_TYPES:
            normalized.pop("expires_at", None)
            return normalized

        ttl = ttl_days
        if ttl is None:
            ttl = DEFAULT_MEMORY_TTL_DAYS_BY_TYPE.get(
                memory_type,
                DEFAULT_MEMORY_TTL_DAYS,
            )
        normalized.setdefault(
            "expires_at",
            (now + timedelta(days=max(1, int(ttl)))).isoformat(),
        )
        return normalized

    @staticmethod
    def _similarity_from_distance(distance: Any) -> float | None:
        if distance is None:
            return None
        try:
            return 1.0 - float(distance)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _lexical_tokens(text: str) -> set[str]:
        normalized = str(text or "").lower()
        chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
        tokens = set(chinese)
        tokens.update(
            "".join(chinese[index : index + 2])
            for index in range(max(len(chinese) - 1, 0))
        )
        tokens.update(re.findall(r"[a-z0-9][a-z0-9_.:/-]*", normalized))
        return {token for token in tokens if token}

    @classmethod
    def _lexical_similarity(cls, query: str, content: str) -> float:
        query_tokens = cls._lexical_tokens(query)
        if not query_tokens:
            return 0.0
        content_tokens = cls._lexical_tokens(content)
        if not content_tokens:
            return 0.0
        return len(query_tokens & content_tokens) / len(query_tokens)

    @staticmethod
    def _hybrid_similarity(
        semantic_similarity: float | None,
        lexical_similarity: float,
    ) -> float:
        lexical = min(max(float(lexical_similarity), 0.0), 1.0)
        if semantic_similarity is None:
            return lexical * 0.9
        semantic = min(max(float(semantic_similarity), 0.0), 1.0)
        return semantic + HYBRID_LEXICAL_WEIGHT * lexical * (1.0 - semantic)

    def _search_collection_lexically(
        self,
        *,
        handler_id: str,
        query: str,
        memory_type: MemoryType | None,
        include_expired: bool,
        active_only: bool,
        n_results: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        where_filter = self._build_where_filter(
            handler_id=handler_id, memory_type=memory_type
        )
        for memory in self._search_lexical_index(
            handler_id=handler_id,
            query=query,
            memory_type=memory_type,
            limit=max(n_results * 4, n_results),
        ):
            score = float(memory.get("lexical_similarity") or 0.0)
            candidates.append(
                {
                    **memory,
                    "distance": 1.0 - score,
                    "_lexical_score": score,
                }
            )
        if self._collection is not None:
            try:
                results = self._collection.get(where=where_filter)
                for index, memory_id in enumerate(results.get("ids") or []):
                    content = str((results.get("documents") or [])[index] or "")
                    raw_metadata = (results.get("metadatas") or [])[index] or {}
                    metadata = self._visible_memory_metadata(raw_metadata)
                    if str(metadata.get("memory_type") or "fact") not in MEMORY_TYPES:
                        continue
                    if not include_expired and self._is_expired_memory(metadata):
                        continue
                    if active_only and self._is_inactive_status_memory(
                        content, metadata
                    ):
                        continue
                    score = self._lexical_similarity(query, content)
                    candidates.append(
                        {
                            "id": memory_id,
                            "content": content,
                            "metadata": metadata,
                            "distance": 1.0 - score,
                            "_lexical_score": score,
                        }
                    )
            except Exception as exc:
                logger.warning("[VectorStore] Failed to scan Chroma memories: %s", exc)

        for memory in self._fallback_memories_for_handler(
            handler_id, memory_type=memory_type
        ):
            content = str(memory.get("content") or "")
            metadata = self._visible_memory_metadata(memory.get("metadata") or {})
            if str(metadata.get("memory_type") or "fact") not in MEMORY_TYPES:
                continue
            if not include_expired and self._is_expired_memory(metadata):
                continue
            if active_only and self._is_inactive_status_memory(content, metadata):
                continue
            score = self._lexical_similarity(query, content)
            candidates.append(
                {
                    **memory,
                    "metadata": metadata,
                    "distance": 1.0 - score,
                    "_lexical_score": score,
                }
            )

        deduplicated: dict[str, dict[str, Any]] = {}
        for memory in candidates:
            memory_id = str(memory.get("id") or "")
            previous = deduplicated.get(memory_id)
            if previous is None or float(memory.get("_lexical_score") or 0.0) > float(
                previous.get("_lexical_score") or 0.0
            ):
                deduplicated[memory_id] = memory

        ranked = list(deduplicated.values())
        ranked.sort(
            key=lambda memory: (
                float(memory.get("_lexical_score") or 0.0),
                str((memory.get("metadata") or {}).get("created_at") or ""),
            ),
            reverse=True,
        )
        selected = ranked[:n_results]
        for memory in selected:
            memory.pop("_lexical_score", None)
            memory.pop("lexical_similarity", None)
        return selected

    def add_memory(
        self,
        handler_id: str,
        content: str,
        memory_type: MemoryType = "fact",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        ttl_days: int | None = None,
        allow_duplicate: bool = False,
        run_maintenance: bool = True,
    ) -> str | None:
        """Add a long-term memory under the ItemHandler scope.

        Memories belong to the handler and are shared by every item it
        drives; they are stored under the handler's id."""
        self._ensure_initialized()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported long-term memory type: {memory_type}")
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        # 写入侧硬上限：记忆是"要点"不是文档。超大内容（测试垃圾、误存的
        # 日志/大文本）截断落盘，防止日后每轮 prompt 被单条记忆撑爆。
        MEMORY_WRITE_MAX_CHARS = 4000
        if len(content) > MEMORY_WRITE_MAX_CHARS:
            logger.warning(
                "[VectorStore] Truncating oversized memory content (%s chars -> %s) for handler %s",
                len(content),
                MEMORY_WRITE_MAX_CHARS,
                handler_id,
            )
            content = content[: MEMORY_WRITE_MAX_CHARS - 20] + "…[truncated]"

        if not allow_duplicate:
            existing = self._check_duplicate(handler_id, content)
            if existing:
                logger.info(
                    f"[VectorStore] Skipping duplicate memory for handler {handler_id}"
                )
                return None

        now = datetime.now()
        meta = self._normalize_memory_lifetime(
            dict(metadata or {}),
            memory_type=memory_type,
            now=now,
            ttl_days=ttl_days,
        )
        meta["handler_id"] = handler_id
        meta["memory_type"] = memory_type
        meta.setdefault("created_at", now.isoformat())

        embedding, used_fallback_embedding = self._encode_single_with_fallback(content)
        if (
            used_fallback_embedding
            or self._collection is None
            or not self._collection_supports_current_model()
        ):
            saved_id = self._add_fallback_memory(
                memory_id=memory_id,
                content=content,
                metadata=meta,
            )
        else:
            self._collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )
            saved_id = memory_id

        logger.info(
            f"[VectorStore] Added {memory_type} memory {memory_id} for handler {handler_id}"
        )
        self._upsert_lexical_memory(
            memory_id=memory_id,
            content=content,
            metadata=meta,
        )
        if run_maintenance:
            self._maybe_compact_expired_memories(handler_id)
        return saved_id

    def _check_duplicate(self, handler_id: str, content: str) -> bool:
        self._ensure_initialized()
        if self._collection is None:
            return any(
                memory.get("content") == content
                for memory in self._fallback_memories_for_handler(handler_id)
            )
        if not self._collection_supports_current_model():
            try:
                results = self._collection.get(
                    where={"handler_id": handler_id},
                    include=["documents"],
                )
                return content in (results.get("documents") or [])
            except Exception:
                return False
        query_embedding = self._try_encode_single(content)
        if query_embedding is None:
            return False

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"handler_id": handler_id},
        )

        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            similarity = 1 - distance
            return similarity >= DEDUP_THRESHOLD

        return False

    def search_memories(
        self,
        handler_id: str,
        query: str,
        n_results: int = 5,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
        active_only: bool = True,
        min_similarity: float | None = None,
        candidate_multiplier: int = DEFAULT_RECALL_CANDIDATE_MULTIPLIER,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self.maintain_memories(handler_id)
        if self._collection is None or not self._collection_supports_current_model():
            return self._search_collection_lexically(
                handler_id=handler_id,
                query=query,
                memory_type=memory_type,
                include_expired=include_expired,
                active_only=active_only,
                n_results=n_results,
            )

        query_embedding = self._try_encode_single(query)
        if query_embedding is None:
            return self._search_collection_lexically(
                handler_id=handler_id,
                query=query,
                memory_type=memory_type,
                include_expired=include_expired,
                active_only=active_only,
                n_results=n_results,
            )
        where_filter = self._build_where_filter(
            handler_id=handler_id, memory_type=memory_type
        )
        query_results = max(n_results, n_results * max(candidate_multiplier, 1))

        memories: list[dict[str, Any]] = []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=query_results,
                where=where_filter,
            )

            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    metadata = self._visible_memory_metadata(metadata)
                    distance = (
                        results["distances"][0][i] if results["distances"] else None
                    )
                    if str(metadata.get("memory_type") or "fact") not in MEMORY_TYPES:
                        continue
                    if not include_expired and self._is_expired_memory(metadata):
                        continue
                    if active_only and self._is_inactive_status_memory(doc, metadata):
                        continue
                    similarity = self._similarity_from_distance(distance)
                    if similarity is None:
                        continue
                    lexical_similarity = self._lexical_similarity(query, doc)
                    hybrid_similarity = self._hybrid_similarity(
                        similarity,
                        lexical_similarity,
                    )
                    memory = {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": metadata,
                        "distance": 1.0 - hybrid_similarity,
                        "_semantic_score": similarity,
                        "_lexical_score": lexical_similarity,
                        "_recall_score": hybrid_similarity,
                    }
                    memories.append(memory)
        except Exception as exc:
            logger.warning("[VectorStore] Failed to search Chroma memories: %s", exc)
            return self._search_collection_lexically(
                handler_id=handler_id,
                query=query,
                memory_type=memory_type,
                include_expired=include_expired,
                active_only=active_only,
                n_results=n_results,
            )

        for memory in self._fallback_memories_for_handler(
            handler_id, memory_type=memory_type
        ):
            metadata = self._visible_memory_metadata(memory.get("metadata", {}))
            content = str(memory.get("content") or "")
            if str(metadata.get("memory_type") or "fact") not in MEMORY_TYPES:
                continue
            if not include_expired and self._is_expired_memory(metadata):
                continue
            if active_only and self._is_inactive_status_memory(content, metadata):
                continue
            lexical_similarity = self._lexical_similarity(query, content)
            hybrid_similarity = self._hybrid_similarity(None, lexical_similarity)
            memories.append(
                {
                    **memory,
                    "metadata": metadata,
                    "distance": 1.0 - hybrid_similarity,
                    "_semantic_score": None,
                    "_lexical_score": lexical_similarity,
                    "_recall_score": hybrid_similarity,
                }
            )

        for memory in self._search_lexical_index(
            handler_id=handler_id,
            query=query,
            memory_type=memory_type,
            limit=query_results,
        ):
            lexical_similarity = float(memory.get("lexical_similarity") or 0.0)
            hybrid_similarity = self._hybrid_similarity(None, lexical_similarity)
            memories.append(
                {
                    **memory,
                    "distance": 1.0 - hybrid_similarity,
                    "_semantic_score": None,
                    "_lexical_score": lexical_similarity,
                    "_recall_score": hybrid_similarity,
                }
            )

        deduplicated: dict[str, dict[str, Any]] = {}
        for memory in memories:
            memory_id = str(memory.get("id") or "")
            previous = deduplicated.get(memory_id)
            if previous is None:
                deduplicated[memory_id] = memory
                continue
            semantic_values = (
                previous.get("_semantic_score"),
                memory.get("_semantic_score"),
            )
            semantic_similarity = max(
                (float(value) for value in semantic_values if value is not None),
                default=None,
            )
            lexical_similarity = max(
                float(previous.get("_lexical_score") or 0.0),
                float(memory.get("_lexical_score") or 0.0),
            )
            hybrid_similarity = self._hybrid_similarity(
                semantic_similarity,
                lexical_similarity,
            )
            preferred = previous
            if len(str(memory.get("content") or "")) > len(
                str(previous.get("content") or "")
            ):
                preferred = memory
            deduplicated[memory_id] = {
                **preferred,
                "distance": 1.0 - hybrid_similarity,
                "_semantic_score": semantic_similarity,
                "_lexical_score": lexical_similarity,
                "_recall_score": hybrid_similarity,
            }
        ranked = sorted(
            (
                memory
                for memory in deduplicated.values()
                if min_similarity is None
                or float(memory.get("_recall_score") or 0.0) >= min_similarity
            ),
            key=lambda memory: float(memory.get("_recall_score") or 0.0),
            reverse=True,
        )[:n_results]
        for memory in ranked:
            memory.pop("_recall_score", None)
            memory.pop("_semantic_score", None)
            memory.pop("_lexical_score", None)
            memory.pop("lexical_similarity", None)
        return ranked

    def get_all_memories(
        self,
        handler_id: str,
        memory_type: MemoryType | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        where_filter = self._build_where_filter(
            handler_id=handler_id, memory_type=memory_type
        )

        memories = []
        if self._collection is not None:
            try:
                results = self._collection.get(where=where_filter)
                if results["ids"]:
                    for i, memory_id in enumerate(results["ids"]):
                        memories.append(
                            {
                                "id": memory_id,
                                "content": results["documents"][i]
                                if results["documents"]
                                else "",
                                "metadata": self._visible_memory_metadata(
                                    results["metadatas"][i]
                                    if results["metadatas"]
                                    else {}
                                ),
                            }
                        )
            except Exception as exc:
                logger.warning("[VectorStore] Failed to read Chroma memories: %s", exc)

        memories.extend(
            self._fallback_memories_for_handler(handler_id, memory_type=memory_type)
        )
        return memories

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        self._ensure_initialized()
        if self._collection is not None:
            try:
                results = self._collection.get(ids=[memory_id])
                if results["ids"]:
                    return {
                        "id": results["ids"][0],
                        "content": results["documents"][0]
                        if results["documents"]
                        else "",
                        "metadata": self._visible_memory_metadata(
                            results["metadatas"][0] if results["metadatas"] else {}
                        ),
                    }
            except Exception as exc:
                logger.warning(
                    "[VectorStore] Failed to read Chroma memory %s: %s", memory_id, exc
                )

        for memory in self._load_fallback_memories():
            if str(memory.get("id") or "") == str(memory_id):
                metadata = (
                    memory.get("metadata")
                    if isinstance(memory.get("metadata"), dict)
                    else {}
                )
                return {
                    "id": str(memory.get("id") or ""),
                    "content": str(memory.get("content") or ""),
                    "metadata": self._visible_memory_metadata(metadata),
                }
        return None

    def get_memory_stats(self, handler_id: str) -> dict[str, Any]:
        all_memories = self.get_all_memories(handler_id)

        stats = {
            "total": 0,
            "by_type": {},
            "expired_count": 0,
        }

        for memory in all_memories:
            meta = memory.get("metadata", {})
            m_type = meta.get("memory_type", "fact")
            if m_type not in MEMORY_TYPES:
                continue
            stats["total"] += 1
            stats["by_type"][m_type] = stats["by_type"].get(m_type, 0) + 1

            if self._is_expired_memory(meta):
                stats["expired_count"] += 1

        return stats

    def delete_memory(self, memory_id: str) -> bool:
        self._ensure_initialized()
        deleted = False
        if self._collection is not None:
            try:
                self._collection.delete(ids=[memory_id])
                logger.info(f"[VectorStore] Deleted memory {memory_id}")
            except Exception as e:
                logger.error(f"[VectorStore] Failed to delete memory {memory_id}: {e}")
            else:
                deleted = True

        fallback_memories = self._load_fallback_memories()
        kept = [
            memory
            for memory in fallback_memories
            if str(memory.get("id") or "") != str(memory_id)
        ]
        if len(kept) != len(fallback_memories):
            self._write_fallback_memories(kept)
            deleted = True
        self._delete_lexical_memories(memory_id=memory_id)
        return deleted

    def delete_handler_memories(self, handler_id: str):
        self._ensure_initialized()
        if self._collection is not None:
            self._collection.delete(where={"handler_id": handler_id})
        fallback_memories = self._load_fallback_memories()
        kept = []
        for memory in fallback_memories:
            metadata = (
                memory.get("metadata")
                if isinstance(memory.get("metadata"), dict)
                else {}
            )
            if str(metadata.get("handler_id") or "") != str(handler_id):
                kept.append(memory)
        if len(kept) != len(fallback_memories):
            self._write_fallback_memories(kept)
        self._delete_lexical_memories(handler_id=handler_id)
        with self._maintenance_lock:
            self._maintenance_items.discard(str(handler_id))
            self._maintenance_last_checked.pop(str(handler_id), None)
        logger.info(f"[VectorStore] Deleted all memories for handler {handler_id}")

    @staticmethod
    def _memory_compaction_scope_key(memory: dict[str, Any]) -> tuple[str, ...]:
        metadata = memory.get("metadata") or {}
        return (
            str(metadata.get("memory_type") or "fact"),
            str(metadata.get("memory_scope") or ""),
            str(metadata.get("robot_id") or ""),
            str(
                metadata.get("robot_conversation_key")
                or metadata.get("conversation_key")
                or ""
            ),
            str(metadata.get("speaker_global_key") or ""),
            str(metadata.get("speaker_key") or ""),
        )

    @staticmethod
    def _memory_compaction_timestamp(memory: dict[str, Any]) -> str:
        metadata = memory.get("metadata") or {}
        return str(metadata.get("updated_at") or metadata.get("created_at") or "")

    @staticmethod
    def _compaction_memory_type(memory_type: str) -> MemoryType:
        if memory_type in MEMORY_TYPES:
            return memory_type  # type: ignore[return-value]
        return "context"

    @staticmethod
    def _compaction_metadata(memories: list[dict[str, Any]]) -> dict[str, Any]:
        source_metadata = memories[0].get("metadata") or {}
        preserved_keys = (
            "memory_scope",
            "robot_id",
            "robot_conversation_key",
            "conversation_key",
            "speaker",
            "speaker_label",
            "speaker_key",
            "speaker_global_key",
        )
        metadata = {
            key: source_metadata[key]
            for key in preserved_keys
            if source_metadata.get(key) not in (None, "")
        }
        metadata.update(
            {
                "type": "expired_memory_compaction",
                "source": "memory_maintenance",
                "compressed_count": len(memories),
                "compressed_from_type": str(
                    source_metadata.get("memory_type") or "fact"
                ),
                "updated_at": datetime.now().isoformat(),
                "verified": all(
                    (entry.get("metadata") or {}).get("verified") is True
                    for entry in memories
                ),
            }
        )
        return metadata

    @staticmethod
    def _compress_expired_memory_content(
        memories: list[dict[str, Any]],
        *,
        memory_type: str,
    ) -> str:
        labels = {
            "fact": "事实",
            "context": "上下文",
            "task": "任务上下文",
        }
        prefix = f"历史{labels.get(memory_type, '长期')}记忆摘要："
        parts: list[str] = []
        seen: set[str] = set()
        for memory in sorted(
            memories,
            key=VectorStoreService._memory_compaction_timestamp,
        ):
            content = re.sub(
                r"\s+",
                " ",
                str(memory.get("content") or "").strip(),
            )
            # 摘要不再进摘要（防"摘要：摘要："嵌套增殖）；传输层格式
            # （CQ 码/消息戳）不是事实，压缩时剥掉而不是带进下一条。
            content = MEMORY_SUMMARY_PREFIX_STRIP_RE.sub("", content)
            content = MEMORY_TRANSPORT_MARKUP_RE.sub(" ", content)
            content = re.sub(r"\s+", " ", content).strip(" ；;")
            if not content:
                continue
            normalized = content.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidate = "；".join([*parts, content])
            if len(prefix) + len(candidate) <= EXPIRED_MEMORY_SUMMARY_MAX_CHARS:
                parts.append(content)
                continue
            remaining = (
                EXPIRED_MEMORY_SUMMARY_MAX_CHARS - len(prefix) - len("；".join(parts))
            )
            if remaining > 24:
                parts.append(content[: max(1, remaining - 2)].rstrip() + "…")
            break
        return prefix + "；".join(parts)

    def _enter_memory_maintenance(self, handler_id: str) -> bool:
        with self._maintenance_lock:
            if handler_id in self._maintenance_items:
                return False
            self._maintenance_items.add(handler_id)
            return True

    def _leave_memory_maintenance(self, handler_id: str) -> None:
        with self._maintenance_lock:
            self._maintenance_items.discard(handler_id)

    def _maybe_compact_expired_memories(self, handler_id: str) -> None:
        try:
            self.maintain_memories(handler_id, interval_seconds=0)
        except Exception as exc:
            logger.warning(
                "[VectorStore] Expired memory maintenance failed for item=%s: %s",
                handler_id,
                exc,
            )

    def maintain_memories(
        self,
        handler_id: str,
        *,
        interval_seconds: float = 300,
    ) -> dict[str, Any]:
        normalized_handler_id = str(handler_id)
        now = time.monotonic()
        with self._maintenance_lock:
            last_checked = self._maintenance_last_checked.get(
                normalized_handler_id, 0.0
            )
            if interval_seconds > 0 and now - last_checked < interval_seconds:
                return {
                    "expired_found": 0,
                    "summarized": 0,
                    "summaries_created": 0,
                }
            self._maintenance_last_checked[normalized_handler_id] = now
        return self.compact_expired_memories(
            normalized_handler_id,
            threshold=EXPIRED_MEMORY_COMPACTION_THRESHOLD,
        )

    def compact_expired_memories(
        self,
        handler_id: str,
        *,
        threshold: int = EXPIRED_MEMORY_COMPACTION_THRESHOLD,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized_handler_id = str(handler_id)
        if not self._enter_memory_maintenance(normalized_handler_id):
            return {
                "expired_found": 0,
                "summarized": 0,
                "summaries_created": 0,
            }

        try:
            by_id: dict[str, dict[str, Any]] = {}
            for memory in self.get_all_memories(normalized_handler_id):
                memory_id = str(memory.get("id") or "")
                metadata = memory.get("metadata") or {}
                if not memory_id or not self._is_expired_memory(metadata):
                    continue
                by_id.setdefault(memory_id, memory)

            expired = list(by_id.values())
            if len(expired) < max(1, int(threshold)):
                return {
                    "expired_found": len(expired),
                    "summarized": 0,
                    "summaries_created": 0,
                }

            groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for memory in expired:
                groups[self._memory_compaction_scope_key(memory)].append(memory)

            summarized = 0
            summaries_created = 0
            for group in groups.values():
                if len(group) < 2:
                    continue
                source_type = str(
                    (group[0].get("metadata") or {}).get("memory_type") or "fact"
                )
                summary_content = self._compress_expired_memory_content(
                    group,
                    memory_type=source_type,
                )
                if not summary_content.strip():
                    continue
                summary_type = self._compaction_memory_type(source_type)
                summary_id = self.add_memory(
                    handler_id=normalized_handler_id,
                    content=summary_content,
                    memory_type=summary_type,
                    metadata=self._compaction_metadata(group),
                    ttl_days=DEFAULT_MEMORY_TTL_DAYS_BY_TYPE.get(summary_type),
                    allow_duplicate=True,
                    run_maintenance=False,
                )
                if not summary_id:
                    continue
                for memory in group:
                    if self.delete_memory(str(memory.get("id") or "")):
                        summarized += 1
                summaries_created += 1

            if summarized:
                logger.info(
                    "[VectorStore] Compacted %s expired memories into %s summaries for handler %s",
                    summarized,
                    summaries_created,
                    normalized_handler_id,
                )
            return {
                "expired_found": len(expired),
                "summarized": summarized,
                "summaries_created": summaries_created,
            }
        finally:
            self._leave_memory_maintenance(normalized_handler_id)

    def expire_old_memories(self, handler_id: str) -> int:
        compaction = self.compact_expired_memories(
            handler_id,
            threshold=EXPIRED_MEMORY_COMPACTION_THRESHOLD,
        )
        expired_ids = {
            str(memory.get("id") or "")
            for memory in self.get_all_memories(handler_id)
            if memory.get("id")
            and self._is_expired_memory(memory.get("metadata") or {})
        }
        for memory_id in expired_ids:
            self.delete_memory(memory_id)
        removed = int(compaction.get("summarized") or 0) + len(expired_ids)
        if removed:
            logger.info(
                "[VectorStore] Removed or compacted %s expired memories for handler %s",
                removed,
                handler_id,
            )
        return removed

    def deduplicate_memories(
        self, handler_id: str, *, threshold: float = DEDUP_THRESHOLD
    ) -> int:
        all_memories = self.get_all_memories(handler_id)
        if len(all_memories) < 2:
            return 0

        contents = [m["content"] for m in all_memories]
        embeddings = self._try_encode(contents)

        mem_types = [
            str((m.get("metadata") or {}).get("memory_type") or "")
            for m in all_memories
        ]

        ids_to_delete = set()
        if embeddings is not None:
            for i in range(len(embeddings)):
                if all_memories[i]["id"] in ids_to_delete:
                    continue
                for j in range(i + 1, len(embeddings)):
                    if all_memories[j]["id"] in ids_to_delete:
                        continue
                    if mem_types[i] != mem_types[j]:
                        continue

                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    if similarity >= threshold:
                        ids_to_delete.add(all_memories[j]["id"])
        else:
            # Embedding model unavailable: fall back to lexical (Jaccard)
            # similarity so near-duplicate cleanup still works.
            logger.info(
                "[VectorStore] Embeddings unavailable; deduplicating handler %s lexically",
                handler_id,
            )
            token_sets = [self._lexical_tokens(content) for content in contents]
            for i in range(len(token_sets)):
                if all_memories[i]["id"] in ids_to_delete or not token_sets[i]:
                    continue
                for j in range(i + 1, len(token_sets)):
                    if all_memories[j]["id"] in ids_to_delete or not token_sets[j]:
                        continue
                    if mem_types[i] != mem_types[j]:
                        continue
                    union = token_sets[i] | token_sets[j]
                    if not union:
                        continue
                    similarity = len(token_sets[i] & token_sets[j]) / len(union)
                    if similarity >= threshold:
                        ids_to_delete.add(all_memories[j]["id"])

        if ids_to_delete:
            if self._collection is not None:
                self._collection.delete(ids=list(ids_to_delete))
            fallback_memories = self._load_fallback_memories()
            kept = [
                memory
                for memory in fallback_memories
                if str(memory.get("id") or "") not in ids_to_delete
            ]
            if len(kept) != len(fallback_memories):
                self._write_fallback_memories(kept)
            logger.info(
                f"[VectorStore] Deduplicated {len(ids_to_delete)} memories for handler {handler_id}"
            )

        return len(ids_to_delete)

    def find_duplicate_clusters(
        self,
        handler_id: str,
        *,
        threshold: float = 0.8,
        max_clusters: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Detect-only near-duplicate clusters (same memory_type, embedding
        similarity >= threshold). Merging is the agent's call via
        compress_memories - this only surfaces suspects."""
        all_memories = self.get_all_memories(handler_id)
        if len(all_memories) < 2:
            return []
        embeddings = self._try_encode([m["content"] for m in all_memories])
        if embeddings is None:
            return []
        mem_types = [
            str((m.get("metadata") or {}).get("memory_type") or "")
            for m in all_memories
        ]
        parent = list(range(len(all_memories)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if mem_types[i] != mem_types[j]:
                    continue
                if self._cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                    parent[find(j)] = find(i)

        groups: dict[int, list[int]] = {}
        for i in range(len(all_memories)):
            groups.setdefault(find(i), []).append(i)
        clusters = [
            [all_memories[i] for i in idxs]
            for idxs in groups.values()
            if len(idxs) >= 2
        ]
        clusters.sort(key=lambda c: -len(c))
        return clusters[:max_clusters]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

    def get_memories_by_type(
        self, handler_id: str, memory_type: MemoryType
    ) -> list[dict[str, Any]]:
        return self.get_all_memories(handler_id, memory_type=memory_type)

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        self._ensure_initialized()
        if self._collection is None:
            return self._update_fallback_memory(
                memory_id, content=content, metadata=metadata
            )

        try:
            existing = self._collection.get(ids=[memory_id])
            if not existing["ids"]:
                return self._update_fallback_memory(
                    memory_id, content=content, metadata=metadata
                )

            current_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            current_doc = existing["documents"][0] if existing["documents"] else ""

            new_content = content or current_doc
            new_meta = {**current_meta, **(metadata or {})}
            if str(new_meta.get("memory_type") or "") in IMMORTAL_MEMORY_TYPES:
                new_meta.pop("expires_at", None)

            embedding, used_fallback_embedding = self._encode_single_with_fallback(
                new_content
            )
            if used_fallback_embedding or not self._collection_supports_current_model():
                if self._collection is not None:
                    self._collection.delete(ids=[memory_id])
                return self._upsert_fallback_memory(
                    memory_id=memory_id,
                    content=new_content,
                    metadata=new_meta,
                )

            self._collection.delete(ids=[memory_id])
            self._collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[new_content],
                metadatas=[new_meta],
            )
            self._upsert_lexical_memory(
                memory_id=memory_id,
                content=new_content,
                metadata=new_meta,
            )

            logger.info(f"[VectorStore] Updated memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"[VectorStore] Failed to update memory {memory_id}: {e}")
            return self._update_fallback_memory(
                memory_id, content=content, metadata=metadata
            )

    def update_memory_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        self._ensure_initialized()
        updates = dict(metadata or {})
        if not updates:
            return False

        updated = False
        if self._collection is not None:
            try:
                existing = self._collection.get(ids=[memory_id])
                if existing["ids"]:
                    current_meta = (
                        existing["metadatas"][0] if existing["metadatas"] else {}
                    )
                    current_doc = (
                        existing["documents"][0] if existing["documents"] else ""
                    )
                    next_meta = {**current_meta, **updates}
                    self._collection.update(ids=[memory_id], metadatas=[next_meta])
                    self._upsert_lexical_memory(
                        memory_id=memory_id,
                        content=current_doc,
                        metadata=next_meta,
                    )
                    updated = True
            except Exception as exc:
                logger.error(
                    "[VectorStore] Failed to update metadata for memory %s: %s",
                    memory_id,
                    exc,
                )

        fallback_updated = self._update_fallback_memory(
            memory_id,
            metadata=updates,
        )
        return updated or fallback_updated

    def supersede_by_memory_key(self, handler_id: str) -> int:
        """Keep only the newest memory per memory_key; delete older same-key entries."""
        all_memories = self.get_all_memories(handler_id)
        if len(all_memories) < 2:
            return 0

        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for memory in all_memories:
            metadata = memory.get("metadata") or {}
            key = str(metadata.get("memory_key") or "").strip()
            if key:
                by_key[key].append(memory)

        ids_to_delete: list[str] = []
        for _key, group in by_key.items():
            if len(group) < 2:
                continue
            group.sort(
                key=lambda m: str(
                    (m.get("metadata") or {}).get("updated_at")
                    or (m.get("metadata") or {}).get("created_at")
                    or ""
                ),
                reverse=True,
            )
            for older in group[1:]:
                older_id = str(older.get("id") or "")
                if older_id:
                    ids_to_delete.append(older_id)

        if not ids_to_delete:
            return 0

        if self._collection is not None:
            self._collection.delete(ids=ids_to_delete)
        for deleted_id in ids_to_delete:
            self._delete_lexical_memories(memory_id=deleted_id)
        fallback_memories = self._load_fallback_memories()
        kept = [
            m
            for m in fallback_memories
            if str(m.get("id") or "") not in set(ids_to_delete)
        ]
        if len(kept) != len(fallback_memories):
            self._write_fallback_memories(kept)

        logger.info(
            "[VectorStore] Superseded %s keyed memories for handler %s",
            len(ids_to_delete),
            handler_id,
        )
        return len(ids_to_delete)

    def get_handler_memory_count(self, handler_id: str) -> int:
        return len(self.get_all_memories(handler_id))

    def summarize_memories(
        self,
        handler_id: str,
        threshold: int = 10,
    ) -> dict[str, Any]:
        result = self.compact_expired_memories(handler_id, threshold=threshold)
        summarized = int(result.get("summarized") or 0)
        summaries_created = int(result.get("summaries_created") or 0)
        expired_found = int(result.get("expired_found") or 0)
        return {
            **result,
            "message": (
                f"压缩了 {summarized} 条过期记忆，生成 {summaries_created} 条摘要"
                if summarized
                else f"过期记忆数量 ({expired_found}) 未达到可压缩条件"
            ),
        }


vector_store = VectorStoreService()
