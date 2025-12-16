"""
Azure Blob Storage Utility for File Uploads

Provides functionality to upload files to Azure Blob Storage and generate
time-limited SAS URLs for secure file access.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from io import BytesIO, StringIO

from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
    ContentSettings,
)
from azure.core.exceptions import AzureError

from config.config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_CONTAINER_NAME,
    AZURE_STORAGE_SAS_EXPIRY_HOURS,
)

logger = logging.getLogger(__name__)


class AzureBlobStorageManager:
    """Manages Azure Blob Storage operations for file uploads"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        """
        Initialize Azure Blob Storage client.
        Allows overriding credentials for specific use cases (e.g., Placestory).
        Falls back to global config if arguments are not provided.
        """
        self.container_name = container_name or AZURE_STORAGE_CONTAINER_NAME

        conn_str = connection_string or AZURE_STORAGE_CONNECTION_STRING

        acc_url = account_url
        acc_key = account_key or AZURE_STORAGE_ACCOUNT_KEY

        acc_name = account_name or AZURE_STORAGE_ACCOUNT_NAME

        if conn_str:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                conn_str
            )
            self.account_name = self._extract_account_name_from_conn_string(conn_str)
            self.account_key = None

        elif acc_url and acc_key:
            self.blob_service_client = BlobServiceClient(
                account_url=acc_url, credential=acc_key
            )
            self.account_name = acc_name or acc_url.split("://")[1].split(".")[0]
            self.account_key = acc_key

        elif acc_name and acc_key:
            constructed_url = f"https://{acc_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=constructed_url, credential=acc_key
            )
            self.account_name = acc_name
            self.account_key = acc_key

        else:
            raise ValueError(
                "Azure Storage credentials not configured. Provide connection_string, account_url+key, or account_name+key."
            )

        self._ensure_container_exists()

    def _extract_account_name_from_conn_string(self, conn_string: str) -> str:
        """Extract account name from connection string"""
        for part in conn_string.split(";"):
            if part.startswith("AccountName="):
                return part.split("=", 1)[1]
        raise ValueError("Could not extract AccountName from connection string")

    def _ensure_container_exists(self):
        """Create container if it doesn't exist with private access (no public access)"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            if not container_client.exists():
                # Create container with private access (no public access)
                container_client.create_container()
                logger.info(
                    f"Created private Azure Blob container: {self.container_name}"
                )
            else:
                logger.debug(f"Azure Blob container exists: {self.container_name}")
        except AzureError as e:
            logger.error(f"Error ensuring container exists: {e}")
            raise

    async def upload_file(
        self,
        file_content: Union[str, bytes, BytesIO, StringIO],
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """
        Upload file to Azure Blob Storage with private access and return pre-signed SAS URL.

        Args:
            file_content: File content as string, bytes, or file-like object
            filename: Name of the file to upload
            content_type: MIME type of the file

        Returns:
            {
                "success": True,
                "blob_url": "https://..." (private, not directly accessible),
                "sas_url": "https://...?sp=r&..." (pre-signed URL with expiry),
                "filename": "...",
                "container": "...",
                "storage_type": "azure_blob",
                "sas_expires_at": "2025-10-09T16:46:00Z"
            }
        """
        try:
            # Convert content to bytes if needed
            if isinstance(file_content, str):
                file_data = file_content.encode("utf-8")
            elif isinstance(file_content, StringIO):
                file_data = file_content.getvalue().encode("utf-8")
            elif isinstance(file_content, BytesIO):
                file_data = file_content.getvalue()
            else:
                file_data = file_content

            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=filename
            )

            # Set content settings
            content_settings = ContentSettings(content_type=content_type)

            # Upload blob (private by default)
            blob_client.upload_blob(
                file_data,
                overwrite=True,
                content_settings=content_settings,
            )

            # Get blob URL (not directly accessible without SAS)
            # Use the blob_client.url which includes the correct filename
            blob_url = blob_client.url
            logger.debug(f"Blob URL generated: {blob_url} (filename: {filename})")

            # Generate pre-signed SAS URL using the actual blob name from the client
            # This ensures the filename matches exactly what was uploaded
            actual_blob_name = blob_client.blob_name
            sas_url = self._generate_sas_url(actual_blob_name)
            expiry_time = datetime.utcnow() + timedelta(
                hours=AZURE_STORAGE_SAS_EXPIRY_HOURS
            )

            logger.info(
                f"Uploaded private file to Azure Blob Storage: {actual_blob_name} "
                f"(SAS expires in {AZURE_STORAGE_SAS_EXPIRY_HOURS} hours). "
                f"URL: {sas_url[:100]}..."
            )

            return {
                "success": True,
                "blob_url": blob_url,
                "sas_url": sas_url,
                "filename": filename,
                "container": self.container_name,
                "storage_type": "azure_blob",
                "sas_expires_at": expiry_time.isoformat() + "Z",
            }

        except AzureError as e:
            logger.error(f"Azure Blob Storage upload error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during blob upload: {e}")
            raise

    def _generate_sas_url(self, blob_name: str) -> str:
        try:
            expiry_time = datetime.utcnow() + timedelta(
                hours=AZURE_STORAGE_SAS_EXPIRY_HOURS
            )

            if self.account_key:
                sas_token = generate_blob_sas(
                    account_name=self.account_name,
                    container_name=self.container_name,
                    blob_name=blob_name,
                    account_key=self.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=expiry_time,
                )
                url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
                logger.debug(f"Generated SAS URL for blob: {blob_name}")
                return url

            url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
            logger.warning(
                f"Generated URL without SAS token (no account key): {blob_name}"
            )
            return url

        except Exception as e:
            logger.error(f"Error generating SAS URL for blob '{blob_name}': {e}")
            # Return URL without SAS token as fallback
            return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"

    def _get_account_key_from_client(self) -> str:
        """Extract account key from blob service client credential"""
        if hasattr(self.blob_service_client, "credential"):
            credential = self.blob_service_client.credential
            if hasattr(credential, "account_key"):
                return credential.account_key
        raise ValueError("Cannot extract account key for SAS generation")


# Singleton instance
_blob_manager = None


def get_blob_manager() -> AzureBlobStorageManager:
    """Get singleton instance of AzureBlobStorageManager"""
    global _blob_manager
    if _blob_manager is None:
        _blob_manager = AzureBlobStorageManager()
    return _blob_manager


async def upload_to_blob_storage(
    file_content: Union[str, bytes, BytesIO, StringIO],
    filename: str,
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    """
    Convenience function to upload file to Azure Blob Storage with private access.

    Args:
        file_content: File content as string, bytes, or file-like object
        filename: Name of the file to upload
        content_type: MIME type of the file

    Returns:
        Dictionary with upload result, blob URL, and pre-signed SAS URL
    """
    manager = get_blob_manager()
    return await manager.upload_file(file_content, filename, content_type)
