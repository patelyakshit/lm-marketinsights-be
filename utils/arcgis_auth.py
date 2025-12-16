import logging
import os
from typing import Dict, Any

from arcgis import GIS
from config.config import ARCGIS_USERNAME, ARCGIS_PASSWORD

from utils.error_handlers import handle_sync_agent_errors, GISError

logger = logging.getLogger(__name__)


class ArcGISAuthManager:
    """Manages ArcGIS authentication and connection"""

    def __init__(self):
        self.api_key = os.getenv("ARCGIS_API_KEY", None)
        self.arcgis_username = ARCGIS_USERNAME or None
        self.arcgis_password = ARCGIS_PASSWORD or None
        self.portal_url = os.getenv("ARCGIS_PORTAL_URL", None)

        self._gis_connection = None
        self._validate_credentials()

        logger.info("ArcGIS Auth Manager initialized")

    def _validate_credentials(self):
        """Validate that all required credentials are present"""
        missing_creds = []

        if self.api_key is None:
            missing_creds.append("ARCGIS_APIKEY")

        if len(missing_creds):
            raise GISError(
                "GIS credentials are missing:" + ", ".join(missing_creds),
                details={
                    "missing": missing_creds
                }
            )

    @handle_sync_agent_errors
    def get_connection(self, force_new: bool = False) -> GIS:
        """Get or create ArcGIS connection"""

        if self._gis_connection is None or force_new:
            try:
                logger.info("Creating new ArcGIS connection")

                # Connect using API key
                if self.api_key is not None:
                    self._gis_connection = GIS(api_key=self.api_key)
                elif self.arcgis_username is not None and self.arcgis_password is not None:
                    self._gis_connection = GIS(
                        self.portal_url, username=self.arcgis_username, password=self.arcgis_password
                    )

                # Test the connection
                self._test_connection()

                logger.info("ArcGIS connection established successfully")
                return self._gis_connection

            except Exception as e:
                logger.error(f"Failed to connect to ArcGIS: {e}")
                raise GISError(
                    f"Authentication failed: {str(e)}",
                    {"error_type": "authentication_failure", "original_error": str(e)},
                )

        return self._gis_connection

    def _test_connection(self):
        """Test the ArcGIS connection by accessing user properties"""
        try:
            # Test connection by getting user info
            if hasattr(self._gis_connection, "users") and hasattr(
                self._gis_connection.users, "me"
            ):
                user_info = self._gis_connection.users.me
                logger.info(
                    f"ArcGIS connection test successful - User: {getattr(user_info, 'username', 'anonymous')}"
                )
            else:
                # For API key connections, test by accessing properties
                properties = self._gis_connection.properties
                if properties:
                    logger.info("ArcGIS connection test successful")
                else:
                    raise GISError(
                        "Connection test failed: Unable to access GIS properties"
                    )
        except Exception as e:
            raise GISError(f"Connection test failed: {str(e)}")

    @handle_sync_agent_errors
    def test_credentials(self) -> Dict[str, Any]:
        """Test ArcGIS credentials and return connection info"""
        try:
            gis = self.get_connection(force_new=True)

            # Get connection properties
            properties = gis.properties

            user_info = None
            try:
                if hasattr(gis, "users") and hasattr(gis.users, "me"):
                    user_info = gis.users.me
            except:
                # API key connections might not have user info
                pass

            return {
                "status": "success",
                "connection_active": True,
                "portal": {
                    "url": getattr(gis, "url", self.portal_url),
                    "name": (
                        properties.get("name", "Unknown") if properties else "Unknown"
                    ),
                    "version": (
                        properties.get("currentVersion", "Unknown")
                        if properties
                        else "Unknown"
                    ),
                },
                "user": {
                    "username": (
                        getattr(user_info, "username", "api_key_user")
                        if user_info
                        else "api_key_user"
                    ),
                    "role": (
                        getattr(user_info, "role", "api_user")
                        if user_info
                        else "api_user"
                    ),
                },
                "connection_type": "api_key",
            }

        except Exception as e:
            logger.error(f"Credential test failed: {e}")
            return {"status": "error", "connection_active": False, "error": str(e)}

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if self._gis_connection:
            return {
                "session_active": True,
                "portal_url": getattr(self._gis_connection, "url", self.portal_url),
                "connection_type": "api_key",
                "version": (
                    getattr(
                        self._gis_connection.properties, "currentVersion", "Unknown"
                    )
                    if hasattr(self._gis_connection, "properties")
                    and self._gis_connection.properties
                    else "Unknown"
                ),
            }
        else:
            return {
                "session_active": False,
                "portal_url": None,
                "connection_type": None,
                "version": None,
            }

    def refresh_connection(self) -> bool:
        """Refresh the ArcGIS connection"""
        try:
            logger.info("Refreshing ArcGIS connection")
            self._gis_connection = None
            self.get_connection(force_new=True)
            return True
        except Exception as e:
            logger.error(f"Failed to refresh connection: {e}")
            return False

    def close_connection(self):
        """Close the current ArcGIS connection"""
        if self._gis_connection:
            try:
                # GIS connection doesn't have explicit close method
                # Just set to None to allow garbage collection
                self._gis_connection = None
                logger.info("ArcGIS connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


# Global instance
arcgis_auth = ArcGISAuthManager()
