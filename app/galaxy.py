import os
import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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


class _AutoRefreshSession(requests.Session):
    """Intercepts all bioblend requests. On 'API key expired' (Galaxy err_code
    401001), refreshes the user's Galaxy API key via the admin key and retries
    the request once with the new key swapped into the URL / headers.
    """

    def __init__(self, owner: "GalaxyClient"):
        super().__init__()
        self._owner = owner
        self._refreshing = False  # prevent recursive refresh

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        if response.status_code == 401 and not self._refreshing:
            try:
                body = response.json()
            except Exception:
                body = {}
            if (body.get("err_code") == 401001
                    and "expired" in body.get("err_msg", "").lower()):
                self._refreshing = True
                try:
                    new_key = self._owner._refresh_user_key()
                    request.url = _replace_key_in_url(request.url, new_key)
                    if "x-api-key" in request.headers:
                        request.headers["x-api-key"] = new_key
                    response = super().send(request, **kwargs)
                except Exception as e:
                    self._owner.logger.warning(f"Auto-refresh failed: {e}")
                finally:
                    self._refreshing = False
        return response


def _replace_key_in_url(url: str, new_key: str) -> str:
    """Swap ?key=... in URL with the new API key (bioblend's auth pattern)."""
    parts = urlparse(url)
    q = parse_qs(parts.query, keep_blank_values=True)
    if "key" in q:
        q["key"] = [new_key]
    return urlunparse(parts._replace(query=urlencode(q, doseq=True)))


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
        # Replace bioblend's session with our auto-refresh session so every
        # bioblend HTTP call goes through the refresh logic.
        self.gi_client.session = _AutoRefreshSession(self)
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

        # Swap the key in-place and rebuild gi_object so subsequent calls use it
        self.user_api_key = new_key
        old_session = self.gi_client.session
        self.gi_object = GalaxyInstance(url=self.galaxy_url, api_key=new_key)
        self.gi_client = self.gi_object.gi
        self.gi_client.session = old_session  # keep the auto-refresh hook
        self.config_client = galaxy.config.ConfigClient(self.gi_client)
        self.logger.info(f"Galaxy API key refreshed for {self.user_email}")
        return new_key

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
