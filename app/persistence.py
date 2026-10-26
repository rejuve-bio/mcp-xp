import os
import re
import logging
from dotenv import load_dotenv
from typing import Any, Optional, List, Iterable

from pymongo import AsyncMongoClient, errors
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.log_setup import configure_logging

load_dotenv()
configure_logging()

class MongoStore:
    
    """
    Async MongoDB key-value store

    Supports flexible connection via URI string or individual parameters.
    Includes retry mechanisms for connection and operations.

    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: str = os.getenv("MONGO_HOST", "localhost"),
        port: int = int(os.getenv("MONGO_PORT", 27017)),
        database_name: str = "Galaxy_Integration",
        username: Optional[str] = None,
        password: Optional[str] = None,
        server_selection_timeout_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
    ):
        
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_ms / 1000.0
        
        self.log = logging.getLogger(__class__.__name__)

        if connection_string:
            self._client = AsyncMongoClient(
                connection_string,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )
        else:
            self._client = AsyncMongoClient(
                host=host,
                port=port,
                username=username,
                password=password,
            )

        self._db = self._client[database_name]
   
    # Lifecycle helpers

    async def __aenter__(self):
        """ Async context manager entry. """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Async context manager exit: close connection. """
        self.close()

    def close(self) -> None:
        """ Cleanly close MongoDB connection. """
        try:
            self._client.close()
            self.log.info("MongoDB connection closed.")
        except Exception as exc:
            self.log.warning(f"Error closing MongoDB connection: {exc}")
            
            
    async def _ping_with_retry(self):
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def ping():
            await self._client.admin.command("ping")

        try:
            await ping()
        except errors.PyMongoError as exc:
            self.log.error("Failed to ping MongoDB after retries.")
            raise RuntimeError("Failed to verify MongoDB connection.") from exc

    async def _ensure_indexes(self, collection_name: str):
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def create():
            await self._db[collection_name].create_index("key", unique=True)
            
        try:
            await create()
        except errors.PyMongoError as exc:
            self.log.error("Failed to create unique index on 'key' after retries.")
            raise RuntimeError("Failed to create index.") from exc

    # Public API
    
    async def verify_connection(self) -> None:
        await self._ping_with_retry()
    
    async def exists(self, collection_name: str, key: str) -> bool:
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(...)
        async def check():
            return await self._db[collection_name].find_one(
                {"key": key}, {"_id": 1}
            ) is not None

        try:
            return await check()
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to check existence of key '{key}'.")
            raise RuntimeError(f"Failed to check key '{key}'.") from exc

    async def set(self, collection_name: str, key: str, value: Any) -> None:
        """ Asynchronously store or update a key-value pair with retry. Uses upsert semantics (idempotent). Supports any JSON-serializable value. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def upsert():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$set": {"value": value}},
                upsert=True,
            )

        try:
            await upsert()
            self.log.debug(f"Successfully set key '{key}'.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to set key '{key}' after retries.")
            raise RuntimeError(f"Failed to store key '{key}'.") from exc

    async def set_if_absent(self, collection_name: str, key: str, value: Any) -> bool:
        """ Store the value only if the key does not exist yet. Returns True if this call stored it.
        Atomic through the unique index on 'key', so concurrent callers cannot both win. """

        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.ConnectionFailure),
            reraise=True,
        )
        async def insert():
            try:
                await self._db[collection_name].insert_one({"key": key, "value": value})
                return True
            except errors.DuplicateKeyError:
                return False

        try:
            return await insert()
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to insert key '{key}' after retries.")
            raise RuntimeError(f"Failed to store key '{key}'.") from exc

    async def get(self, collection_name: str, key: str) -> Optional[Any]:
        """ Asynchronously retrieve a value by key with retry. Returns None if key does not exist. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch():
            doc = await self._db[collection_name].find_one(
                {"key": key},
                {"_id": 0, "value": 1},
            )
            return doc["value"] if doc else None

        try:
            return await fetch()
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to fetch key '{key}' after retries.")
            raise RuntimeError(f"Failed to fetch key '{key}'.") from exc
        
    async def extend(self, collection_name: str, key: str, new_values: list[Any]) -> None:
        """ Asynchronously extend the list under the key with new_values. Creates the key if it doesn't exist (idempotent). Assumes the value is a list; supports any JSON-serializable values. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")
        
        if not isinstance(new_values, list):
            raise ValueError("new_values must be a list.")
        
        if not new_values:  # Nothing to add
            self.log.debug(f"No new values to extend for key '{key}'.")
            return
        
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$push": {"value": {"$each": new_values}}},
                upsert=True,
            )
        
        try:
            await update()
            self.log.debug(f"Successfully extended key '{key}' with {len(new_values)} new values.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to extend key '{key}' after retries.")
            raise RuntimeError(f"Failed to extend key '{key}'.") from exc
        
    async def delete_specific(self, collection_name: str, key: str, elements: Iterable[Any]) -> None:
        """ Asynchronously remove elements from the list under the key where the 'id' field matches any of the provided ids. Does nothing if the key or matching elements don't exist. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")
                
        if not elements:  # Nothing to remove
            self.log.debug(f"No ids provided to delete for key '{key}'.")
            return
        
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            result = await self._db[collection_name].update_one(
                {"key": key},
                {"$pull": {"value": {"id": {"$in": elements}}}},
            )
            return result
        
        try:
            result = await update()
            if result.modified_count > 0:
                self.log.debug(f"Successfully removed matching element(s) from key '{key}'.")
            else:
                self.log.debug(f"No matching elements found to remove from key '{key}'.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to delete from key '{key}' after retries.")
            raise RuntimeError(f"Failed to delete from key '{key}'.") from exc

    async def add_to_set(self, collection_name: str, key: str, elements: Iterable[Any]) -> None:
        """ Atomically add one or more elements to a set-like array value. """

        await self._ensure_indexes(collection_name)
         
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        elements = list(elements)
        if not elements:
            return  # no-op by design

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$addToSet": {"value": {"$each": elements}}},
                upsert=True,
            )

        try:
            await update()
            self.log.debug(f"Added {len(elements)} element(s) to set '{key}'.")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to add elements to set '{key}' after retries.")
            raise RuntimeError(f"Failed to update set for key '{key}'.") from exc
    
    async def delete(self, collection_name: str, key: str) -> bool:
        """ Asynchronously delete a key-value pair if it exists. Returns True if deleted, False if key did not exist. """
        
        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def remove():
            result = await self._db[collection_name].delete_one({"key": key})
            return result.deleted_count > 0

        try:
            deleted = await remove()
            if deleted:
                self.log.debug(f"Successfully deleted key '{key}'.")
                return True
            else:
                return False
                
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to delete key '{key}' after retries.")
            raise RuntimeError(f"Failed to delete key '{key}'.") from exc       
        
    async def list_keys(self, collection_name: str, prefix: Optional[str] = None, limit: int = 100) -> List[str]:
        """ Asynchronously list keys, optionally filtered by prefix. """
        
        await self._ensure_indexes(collection_name)
         
        query = {"key": {"$regex": f"^{prefix}"}} if prefix else {}
        
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch_keys():
            keys = []
            cursor = self._db[collection_name].find(query, {"_id": 0, "key": 1}).limit(limit)
            async for doc in cursor:
                keys.append(doc["key"])
            return keys

        try:
            return await fetch_keys()
        except errors.PyMongoError as exc:
            self.log.error("Failed to list keys after retries.")
            raise RuntimeError("Failed to list keys.") from exc
            
    async def update_value_element(
        self,
        collection_name: str,
        key: str,
        match_field: str,
        match_value: Any,
        update_field: str,
        new_value: Any,
    ) -> None:
        """ Update a field inside a matched object in the `value` array. """

        await self._ensure_indexes(collection_name)

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            await self._db[collection_name].update_one(
                {
                    "key": key,
                    f"value.{match_field}": match_value,
                },
                {
                    "$set": {
                        f"value.$.{update_field}": new_value
                    }
                },
            )

        try:
            await update()
        except errors.PyMongoError as exc:
            self.log.error("Failed to update a selected element.")
            raise RuntimeError("Failed to update a selected element.") from exc
        

    async def remove_from_set(self, collection_name: str, key: str, elements: Iterable[Any]) -> int:
        """ Atomically remove one or more elements from the set-like array value. """

        await self._ensure_indexes(collection_name)
        
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        elements = list(elements)
        if not elements:
            return 0  # no-op by design

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def update():
            result = await self._db[collection_name].update_one(
                {"key": key},
                {"$pull": {"value": {"$in": elements}}},
            )
            return result.modified_count

        try:
            modified = await update()
            self.log.debug(f"Removed elements from set '{key}': modified {modified}.")
            return modified
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to remove elements from set '{key}' after retries.")
            raise RuntimeError(f"Failed to update set for key '{key}'.") from exc
        
    async def remove_from_value_array(
        self,
        collection_name: str,
        key: str,
        match_field: str,
        match_value: Any,
    ) -> bool:
        """ Remove element(s) from the `value` array where match_field == match_value. Returns True if at least one element was removed. """

        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def pull():
            result = await self._db[collection_name].update_one(
                {"key": key},
                {
                    "$pull": {
                        "value": {match_field: match_value}
                    }
                },
            )
            return result.modified_count > 0

        try:
            removed = await pull()
            if removed:
                self.log.debug(
                    f"Removed element(s) from value array where {match_field}={match_value}."
                )
            return removed
        except errors.PyMongoError as exc:
            self.log.error("Failed to remove element(s) from value array after retries.")
            raise RuntimeError("Failed to update value array.") from exc
        
    async def set_element(self, collection_name: str, key: str, field: str, value: Any) -> None:
        """ Asynchronously set a field in a hash-like document. """
        
        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Hash key must be a non-empty string.")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Field must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def upsert():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$set": {f"value.{field}": value}},
                upsert=True,
            )

        try:
            await upsert()
            self.log.debug(f"set to {collection_name}:{key} {field}")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to set {collection_name}:{key}:{field} - {exc}")

    async def get_element(self, collection_name: str, key: str, field: str) -> Optional[Any]:
        """ Asynchronously get a single field from a document. """
        
        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Hash key must be a non-empty string.")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Field must be a non-empty string.")

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch():
            doc = await self._db[collection_name].find_one(
                {"key": key},
                {"_id": 0, "value": 1},
            )
            value_dict = doc.get("value") if doc else None
            if isinstance(value_dict, dict):
                return value_dict.get(field)
            return None

        try:
            return await fetch()
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to get {collection_name}:{key}:{field} - {exc}")
            return None
        
    async def delete_elements(self, collection_name: str, key: str, fields: Iterable[str]) -> None:
        """ Asynchronously delete multiple fields from a hash-like document. """
        
        await self._ensure_indexes(collection_name)

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Hash key must be a non-empty string.")
        if not isinstance(fields, list) or not all(isinstance(f, str) and f.strip() for f in fields):
            raise ValueError("Fields must be a list of non-empty strings.")

        # Construct the $unset dict dynamically
        unset_dict = {f"value.{f}": "" for f in fields}

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def unset():
            await self._db[collection_name].update_one(
                {"key": key},
                {"$unset": unset_dict},
            )

        try:
            await unset()
            self.log.debug(f"Deleted fields {fields} from {collection_name}:{key}")
        except errors.PyMongoError as exc:
            self.log.error(f"Failed to delete fields {fields} from {collection_name}:{key} - {exc}")

    async def find_by_value_field(
        self,
        collection_name: str,
        field_path: str,
        value: Any,
        key_prefix: Optional[str] = None,
        limit: int = 500,
    ) -> List[dict]:
        """Return `[{key, value}, ...]` for documents whose nested `value.<field_path>` equals `value`.

        When `key_prefix` is provided, only docs whose `key` starts with `{key_prefix}:`
        are returned — used to scope reverse lookups by username so cross-user reuse
        of the same field value cannot leak records across users.
        """

        await self._ensure_indexes(collection_name)

        query: dict = {f"value.{field_path}": value}
        if key_prefix:
            if not isinstance(key_prefix, str) or not key_prefix.strip():
                raise ValueError("key_prefix must be a non-empty string when provided.")
            query["key"] = {"$regex": f"^{re.escape(key_prefix)}:"}

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_fixed(self._retry_delay_seconds),
            retry=retry_if_exception_type(errors.PyMongoError),
            reraise=True,
        )
        async def fetch():
            results: List[dict] = []
            cursor = self._db[collection_name].find(
                query, {"_id": 0, "key": 1, "value": 1}
            ).limit(limit)
            async for doc in cursor:
                results.append(doc)
            return results

        try:
            return await fetch()
        except errors.PyMongoError as exc:
            self.log.error(
                f"Failed to find by value field '{field_path}' in {collection_name} after retries."
            )
            raise RuntimeError(
                f"Failed to find by value field '{field_path}'."
            ) from exc