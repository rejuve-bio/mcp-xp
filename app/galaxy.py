import os
import json
import logging
from typing import Optional

import requests
from dotenv import load_dotenv
from bioblend import galaxy
from bioblend.galaxy.objects import GalaxyInstance
from bioblend.galaxy.client import ConnectionError as GalaxyConnectionError
from requests.exceptions import RequestException

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
    )

from app.auth import galaxy_accounts
from app.context import current_user_email

load_dotenv()


# bioblend's request methods; each sends the key from gi_client.json_headers.
_REQUEST_METHODS = (
    "make_get_request", "make_post_request", "make_put_request",
    "make_delete_request", "make_patch_request",
)


def _is_expired_key(status_code, body) -> bool:
    """Galaxy's answer for a key that is no longer the user's newest one."""
    if status_code != 401:
        return False
    try:
        data = json.loads(body) if isinstance(body, (str, bytes)) else body
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("err_code") == 401001


def _with_key_recovery(owner: "GalaxyClient", method):
    """Retry a bioblend request once with the user's current key when Galaxy
    says the key expired. GET/DELETE return the response; the others raise."""
    def call(*args, **kwargs):
        try:
            result = method(*args, **kwargs)
        except GalaxyConnectionError as e:
            if _is_expired_key(e.status_code, e.body) and owner._recover_key():
                return method(*args, **kwargs)
            raise
        if (isinstance(result, requests.Response)
                and _is_expired_key(result.status_code, result.content)
                and owner._recover_key()):
            return method(*args, **kwargs)
        return result
    return call


class GalaxyClient:
    """Client wrapper for interacting with a Galaxy instance (production-ready)."""

    def __init__(
        self,
        user_api_key: str,
        galaxy_url: Optional[str] = None,
        user_email: Optional[str] = None,
    ):
        # Load configuration
        self.galaxy_url: str = galaxy_url or os.getenv("GALAXY_URL", "")
        self.admin_api_key: Optional[str] = os.getenv("GALAXY_API_KEY")
        self.user_api_key: str = user_api_key
        self.user_email: Optional[str] = user_email or current_user_email.get(None)

        # Retry and logger config
        self.max_retries: int = 3
        self._key_recovered = False
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.galaxy_url:
            raise ValueError("GALAXY_URL is not set in environment or passed explicitly.")
        if not self.user_api_key:
            raise ValueError("User API key must be provided to GalaxyClient.")
        if not self.admin_api_key:
            self.logger.warning("GALAXY_API_KEY (admin) is not set. Admin functionalities will fail.")
        if not self.user_email:
            self.logger.warning("user_email not provided; auto-refresh of expired keys will be disabled.")

        try:
            self._init_gi()
            self.logger.info(
                "GalaxyClient initialized for %s (retries=%s)",
                self.galaxy_url, self.max_retries
            )
        except GalaxyConnectionError as e:
            self.logger.error(f"Failed to connect to Galaxy at {self.galaxy_url}: {e}")
            raise GalaxyConnectionError(f"Failed to connect to Galaxy at {self.galaxy_url}: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error initializing GalaxyClient: {e}")
            raise Exception(f"Unexpected error initializing GalaxyClient: {e}") from e

    def _init_gi(self):
        """Build admin + user GalaxyInstance and wire the auto-refresh session."""
        self.gi_admin: Optional[GalaxyInstance] = (
            GalaxyInstance(url=self.galaxy_url, api_key=self.admin_api_key)
            if self.admin_api_key else None
        )
        self.gi_object: GalaxyInstance = GalaxyInstance(
            url=self.galaxy_url, api_key=self.user_api_key
        )
        self.gi_client = self.gi_object.gi
        # bioblend sends every request through these methods (with plain
        # `requests`, not a session), so recovery from an expired key wraps them.
        for name in _REQUEST_METHODS:
            setattr(self.gi_client, name, _with_key_recovery(self, getattr(self.gi_client, name)))
        self.config_client = galaxy.config.ConfigClient(self.gi_client)

    def _refresh_user_key(self) -> str:
        """Switch to the user's current Galaxy key after Galaxy reported this one expired.

        Reads the newest key with the admin key instead of minting another, so
        it doesn't expire the key held by the user's other sessions. Needs
        user_email, which is set only from a verified login token.
        """
        if not self.user_email or not self.admin_api_key:
            raise RuntimeError(
                "Cannot auto-refresh API key: user_email or admin_api_key missing"
            )
        self.logger.info(f"Refreshing expired Galaxy API key for {self.user_email}")
        galaxy_accounts.forget_key(self.user_email)
        new_key, _ = galaxy_accounts.key_for_email(self.user_email, create=False)
        if not new_key:
            raise RuntimeError(f"No Galaxy user found for email {self.user_email}")

        # Swap the key in place: managers built on this client keep working.
        self.user_api_key = new_key
        self.gi_client._key = new_key
        self.gi_client.json_headers["x-api-key"] = new_key
        self.logger.info(f"Galaxy API key refreshed for {self.user_email}")
        return new_key

    def _recover_key(self) -> bool:
        """Switch to the current key once per client; False if that isn't possible."""
        if self._key_recovered:
            return False
        self._key_recovered = True
        try:
            self._refresh_user_key()
            return True
        except Exception as e:
            self.logger.warning(f"Auto-refresh failed: {e}")
            return False

    @property
    def whoami(self) -> str:
        """
        Return current user info with retries, error handling, and safe fallback.
        Retries up to 3 times with exponential backoff on network errors.
        """
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((GalaxyConnectionError, RequestException)),
            reraise=True,
        )
        def _whoami():
            try:
                whoami = self.config_client.whoami()
                if not whoami or "id" not in whoami:
                    self.logger.error("Received invalid response from Galaxy whoami: %s", whoami)
                    return {"error": "Invalid whoami response"}
                return whoami.get("username")
            except (GalaxyConnectionError, RequestException) as e:
                error_msg = str(e) if not isinstance(e, str) else e
                if "API key has expired" in error_msg:
                    self.logger.error("Network error fetching user identity: API key expired. Re-authentication required to refresh the token.")
                else:
                    self.logger.error("Network error fetching user identity: %s", e)
                raise
            except Exception as e:
                self.logger.exception("Unexpected error in whoami: %s", e)
                raise

        return _whoami()

    def validate_connection(self) -> bool:
        """
        Check if connection to Galaxy is alive.
        Returns True if valid, False otherwise.
        """
        try:
            whoami = self.whoami
            if isinstance(whoami, dict) and "id" in whoami:
                self.logger.debug("Connection valid: %s", whoami)
                return True
            self.logger.warning("Connection validation failed: response missing id.")
            return False
        except Exception as e:
            self.logger.warning("Validation failed: %s", e)
            return False
