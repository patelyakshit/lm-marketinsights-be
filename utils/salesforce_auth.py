import logging
import os
from typing import Dict, Any

from simple_salesforce import Salesforce

from utils.error_handlers import SalesforceError, handle_sync_agent_errors

logger = logging.getLogger(__name__)


class SalesforceAuthManager:
    """Manages Salesforce authentication and connection"""

    def __init__(self):
        self.username = os.getenv("SALESFORCE_USERNAME")
        self.password = os.getenv("SALESFORCE_PASSWORD")
        self.security_token = os.getenv("SALESFORCE_SECURITY_TOKEN")
        self.domain = os.getenv("SALESFORCE_DOMAIN")
        self.instance_url = os.getenv("SALESFORCE_INSTANCE_URL")

        self._sf_connection = None
        self._validate_credentials()

        logger.info("Salesforce Auth Manager initialized")

    def _validate_credentials(self):
        """Validate that all required credentials are present"""
        missing_creds = []

        if not self.username:
            missing_creds.append("SALESFORCE_USERNAME")
        if not self.password:
            missing_creds.append("SALESFORCE_PASSWORD")
        if not self.security_token:
            missing_creds.append("SALESFORCE_SECURITY_TOKEN")

        if missing_creds:
            raise SalesforceError(
                f"Missing required Salesforce credentials: {', '.join(missing_creds)}",
                {"missing_credentials": missing_creds},
            )

    @handle_sync_agent_errors
    def get_connection(self, force_new: bool = False) -> Salesforce:
        """Get or create Salesforce connection"""
        if self._sf_connection is None or force_new:
            try:
                logger.info("Creating new Salesforce connection")

                connection_params = {
                    "username": self.username,
                    "password": self.password,
                    "security_token": self.security_token,
                }

                # Add domain if specified
                if self.domain:
                    connection_params["domain"] = self.domain

                # Add instance URL if specified
                if self.instance_url:
                    connection_params["instance_url"] = self.instance_url

                self._sf_connection = Salesforce(**connection_params)

                # Test the connection
                self._test_connection()

                logger.info("Salesforce connection established successfully")
                return self._sf_connection

            except Exception as e:
                logger.error(f"Failed to connect to Salesforce: {e}")
                raise SalesforceError(
                    f"Authentication failed: {str(e)}",
                    {"error_type": "authentication_failure", "original_error": str(e)},
                )

        return self._sf_connection

    def _test_connection(self):
        """Test the Salesforce connection by making a simple query"""
        try:
            # Try to get org info to verify connection
            result = self._sf_connection.query("SELECT Id FROM Organization LIMIT 1")
            if not result.get("records"):
                raise SalesforceError("Connection test failed: No organization found")
            logger.info("Salesforce connection test successful")
        except Exception as e:
            raise SalesforceError(f"Connection test failed: {str(e)}")

    @handle_sync_agent_errors
    def test_credentials(self) -> Dict[str, Any]:
        """Test Salesforce credentials and return connection info"""
        try:
            sf = self.get_connection(force_new=True)

            # Get organization info
            org_query = "SELECT Id, Name, OrganizationType, InstanceName FROM Organization LIMIT 1"
            org_result = sf.query(org_query)

            if org_result["records"]:
                org_info = org_result["records"][0]

                return {
                    "status": "success",
                    "connection_active": True,
                    "organization": {
                        "id": org_info.get("Id"),
                        "name": org_info.get("Name"),
                        "type": org_info.get("OrganizationType"),
                        "instance": org_info.get("InstanceName"),
                    },
                    "session_id": sf.session_id,
                    "instance_url": sf.base_url,
                }
            else:
                return {
                    "status": "error",
                    "connection_active": False,
                    "error": "No organization found",
                }

        except Exception as e:
            logger.error(f"Credential test failed: {e}")
            return {"status": "error", "connection_active": False, "error": str(e)}

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if self._sf_connection:
            return {
                "session_active": True,
                "session_id": self._sf_connection.session_id,
                "instance_url": self._sf_connection.base_url,
                "api_version": self._sf_connection.api_version,
            }
        else:
            return {
                "session_active": False,
                "session_id": None,
                "instance_url": None,
                "api_version": None,
            }

    def refresh_connection(self) -> bool:
        """Refresh the Salesforce connection"""
        try:
            logger.info("Refreshing Salesforce connection")
            self._sf_connection = None
            self.get_connection(force_new=True)
            return True
        except Exception as e:
            logger.error(f"Failed to refresh connection: {e}")
            return False

    def close_connection(self):
        """Close the current Salesforce connection"""
        if self._sf_connection:
            try:
                # simple_salesforce doesn't have an explicit close method
                # Just set to None to allow garbage collection
                self._sf_connection = None
                logger.info("Salesforce connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


# Global instance
salesforce_auth = SalesforceAuthManager()
