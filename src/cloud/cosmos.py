"""Cosmos DB (NoSQL API) over REST with master-key auth: upsert, query by partition, delete.

Documents are partitioned by ``owner`` (an anonymous ID, or a hashed Entra user ID), so every
read and delete touches exactly one partition.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.parse
from email.utils import formatdate
from typing import Any, Dict, List

from src.cloud.http import json_body, request

API_VERSION = "2018-12-31"


def _auth(key: str, verb: str, resource_type: str, resource_link: str, date: str) -> str:
    text = f"{verb.lower()}\n{resource_type.lower()}\n{resource_link}\n{date.lower()}\n\n"
    signature = base64.b64encode(hmac.new(base64.b64decode(key), text.encode("utf-8"), hashlib.sha256).digest()).decode()
    return urllib.parse.quote(f"type=master&ver=1.0&sig={signature}", safe="")


class CosmosContainer:
    def __init__(self, endpoint: str, key: str, database: str, container: str) -> None:
        self.endpoint = endpoint.rstrip("/") + "/"
        self.key = key
        self.collection_link = f"dbs/{database}/colls/{container}"

    def _headers(self, verb: str, resource_link: str, partition: str, extra: Dict[str, str] | None = None) -> Dict[str, str]:
        date = formatdate(usegmt=True)
        headers = {
            "Authorization": _auth(self.key, verb, "docs", resource_link, date),
            "x-ms-date": date,
            "x-ms-version": API_VERSION,
            "x-ms-documentdb-partitionkey": json.dumps([partition]),
            "Accept": "application/json",
        }
        headers.update(extra or {})
        return headers

    def upsert(self, document: Dict[str, Any], partition: str) -> None:
        headers = self._headers("POST", self.collection_link, partition,
                                {"Content-Type": "application/json", "x-ms-documentdb-is-upsert": "True"})
        request("cosmos", "POST", self.endpoint + self.collection_link + "/docs", headers, json_body(document))

    def query(self, sql: str, parameters: List[Dict[str, Any]], partition: str) -> List[Dict[str, Any]]:
        headers = self._headers("POST", self.collection_link, partition, {
            "Content-Type": "application/query+json",
            "x-ms-documentdb-isquery": "True",
            "x-ms-max-item-count": "100",
        })
        _, payload = request("cosmos", "POST", self.endpoint + self.collection_link + "/docs", headers,
                             json_body({"query": sql, "parameters": parameters}))
        return list((payload or {}).get("Documents") or [])

    def delete(self, document_id: str, partition: str) -> None:
        link = f"{self.collection_link}/docs/{document_id}"
        request("cosmos", "DELETE", self.endpoint + urllib.parse.quote(link), self._headers("DELETE", link, partition))
