"""
Store Knowledge Service

Provides fast access to user's store/location data for AI context.
Supports multiple data sources:
1. JSON cache (fast reads)
2. Salesforce sync (enterprise integration)
3. CSV upload (self-service)

Each organization has its own store data, isolated by org_id.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Path to store data directory
KNOWLEDGE_DIR = Path(__file__).parent
STORES_DIR = KNOWLEDGE_DIR / "stores"


class StoreKnowledgeService:
    """
    Fast access to organization's store/location data.

    Features:
    - Instant lookups from JSON cache (< 1ms)
    - Organization-isolated data (multi-tenant)
    - Multiple store identifiers (ID, name, store number)
    - Location data for AI context
    """

    def __init__(self, org_id: str = "default"):
        """
        Initialize the service for a specific organization.

        Args:
            org_id: Organization identifier for multi-tenant isolation
        """
        self.org_id = org_id
        self._stores: Dict[str, Dict] = {}
        self._stores_by_name: Dict[str, str] = {}  # name -> store_id mapping
        self._stores_by_number: Dict[str, str] = {}  # store_number -> store_id mapping
        self._loaded = False
        self._json_path = STORES_DIR / f"{org_id}_stores.json"
        self._metadata: Dict[str, Any] = {}

        # Ensure stores directory exists
        STORES_DIR.mkdir(parents=True, exist_ok=True)

        # Load data on init
        self._load_data()

    def _load_data(self) -> None:
        """Load store data from JSON file."""
        try:
            if self._json_path.exists():
                with open(self._json_path, "r") as f:
                    data = json.load(f)

                self._metadata = data.get("metadata", {})
                stores_list = data.get("stores", [])

                # Build lookup indexes
                for store in stores_list:
                    store_id = store.get("id") or store.get("store_id")
                    if store_id:
                        self._stores[str(store_id)] = store

                        # Index by name (lowercase for case-insensitive lookup)
                        if store.get("name"):
                            self._stores_by_name[store["name"].lower()] = str(store_id)

                        # Index by store number
                        if store.get("store_number"):
                            self._stores_by_number[str(store["store_number"])] = str(store_id)

                self._loaded = True
                logger.info(
                    f"Loaded {len(self._stores)} stores for org '{self.org_id}'"
                )
            else:
                logger.info(f"No store data found for org '{self.org_id}'")
        except Exception as e:
            logger.error(f"Error loading store data for org '{self.org_id}': {e}")

    def get_store(self, identifier: str) -> Optional[Dict]:
        """
        Get store by ID, name, or store number.

        Args:
            identifier: Store ID, name, or number (e.g., "14", "Store #14", "Downtown Store")

        Returns:
            Store dict with all metadata, or None if not found
        """
        identifier = str(identifier).strip()

        # Try direct ID lookup first
        if identifier in self._stores:
            return self._stores[identifier]

        # Try store number (remove "store", "#", etc.)
        clean_number = identifier.lower().replace("store", "").replace("#", "").replace(" ", "").strip()
        if clean_number in self._stores_by_number:
            store_id = self._stores_by_number[clean_number]
            return self._stores.get(store_id)

        # Try name lookup (case-insensitive)
        name_lower = identifier.lower()
        if name_lower in self._stores_by_name:
            store_id = self._stores_by_name[name_lower]
            return self._stores.get(store_id)

        # Fuzzy match on name
        for name, store_id in self._stores_by_name.items():
            if name_lower in name or name in name_lower:
                return self._stores.get(store_id)

        return None

    def get_store_response(self, identifier: str) -> str:
        """
        Get a formatted response for a store query.

        Args:
            identifier: Store ID, name, or number

        Returns:
            Formatted markdown response
        """
        store = self.get_store(identifier)

        if not store:
            return self._unknown_store_response(identifier)

        # Build formatted response
        name = store.get("name", "Unknown Store")
        store_number = store.get("store_number", "N/A")
        address = store.get("address", "")
        city = store.get("city", "")
        state = store.get("state", "")
        zip_code = store.get("zip", "")
        latitude = store.get("latitude")
        longitude = store.get("longitude")

        # Additional fields
        phone = store.get("phone", "")
        manager = store.get("manager", "")
        revenue = store.get("revenue", "")
        customer_count = store.get("customer_count", "")
        status = store.get("status", "Active")

        # Format full address
        full_address = address
        if city:
            full_address += f", {city}"
        if state:
            full_address += f", {state}"
        if zip_code:
            full_address += f" {zip_code}"

        response = f"""**{name}** (Store #{store_number})

**Location:**
- Address: {full_address}
- Coordinates: {latitude}, {longitude}

**Details:**
- Status: {status}"""

        if phone:
            response += f"\n- Phone: {phone}"
        if manager:
            response += f"\n- Manager: {manager}"
        if customer_count:
            response += f"\n- Customer Count: {customer_count:,}" if isinstance(customer_count, int) else f"\n- Customer Count: {customer_count}"
        if revenue:
            response += f"\n- Revenue (12-month): ${revenue:,.0f}" if isinstance(revenue, (int, float)) else f"\n- Revenue: {revenue}"

        return response

    def get_store_for_analysis(self, identifier: str) -> Optional[Dict]:
        """
        Get store data optimized for GIS/location analysis.

        Returns store with coordinates for trade area analysis.
        """
        store = self.get_store(identifier)
        if not store:
            return None

        return {
            "name": store.get("name"),
            "store_number": store.get("store_number"),
            "address": self._build_full_address(store),
            "latitude": store.get("latitude"),
            "longitude": store.get("longitude"),
            "city": store.get("city"),
            "state": store.get("state"),
        }

    def _build_full_address(self, store: Dict) -> str:
        """Build full address string from store data."""
        parts = [
            store.get("address", ""),
            store.get("city", ""),
            store.get("state", ""),
            store.get("zip", ""),
        ]
        return ", ".join(p for p in parts if p)

    def get_all_stores(self) -> List[Dict]:
        """Get all stores for the organization."""
        return list(self._stores.values())

    def get_all_stores_summary(self) -> str:
        """Get a summary of all stores."""
        if not self._loaded or not self._stores:
            return f"No stores found for this organization. You can add stores via CSV upload or Salesforce sync."

        total = len(self._stores)
        stores_list = []
        for store_id, store in list(self._stores.items())[:20]:  # Show first 20
            name = store.get("name", "Unknown")
            number = store.get("store_number", store_id)
            city = store.get("city", "")
            stores_list.append(f"- **Store #{number}**: {name} ({city})")

        response = f"""**Your Stores** ({total} total)

{chr(10).join(stores_list)}"""

        if total > 20:
            response += f"\n\n...and {total - 20} more stores."

        response += "\n\nAsk about any store by number (e.g., 'What is store 14?') or name!"

        return response

    def search_stores(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search stores by keyword.

        Args:
            query: Search query (name, city, state, etc.)
            limit: Max results

        Returns:
            List of matching stores
        """
        query_lower = query.lower()
        results = []

        for store_id, store in self._stores.items():
            score = 0

            # Check name
            if query_lower in store.get("name", "").lower():
                score += 10

            # Check city
            if query_lower in store.get("city", "").lower():
                score += 5

            # Check state
            if query_lower in store.get("state", "").lower():
                score += 3

            # Check address
            if query_lower in store.get("address", "").lower():
                score += 2

            if score > 0:
                results.append({"store": store, "score": score})

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["store"] for r in results[:limit]]

    def _unknown_store_response(self, identifier: str) -> str:
        """Response for unknown store."""
        store_count = len(self._stores)

        if store_count == 0:
            return f"""I don't have any store data loaded for your organization yet.

**To add your stores:**
1. **CSV Upload** - Upload a CSV file with store locations
2. **Salesforce Sync** - Connect your Salesforce Account/Store objects

Contact your administrator to set up store data."""

        return f"""I couldn't find a store matching **"{identifier}"**.

You have **{store_count} stores** in your organization.

Try:
- Store number (e.g., "store 14" or just "14")
- Store name (e.g., "Downtown Store")
- "List all stores" to see your stores

Or ask: "What stores are in Dallas?" to search by city."""

    def save_stores(self, stores: List[Dict], source: str = "manual") -> bool:
        """
        Save stores to JSON file.

        Args:
            stores: List of store dictionaries
            source: Data source (csv, salesforce, manual)

        Returns:
            True if saved successfully
        """
        try:
            data = {
                "metadata": {
                    "org_id": self.org_id,
                    "source": source,
                    "updated_at": datetime.utcnow().isoformat(),
                    "store_count": len(stores),
                },
                "stores": stores,
            }

            with open(self._json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Reload data
            self._stores.clear()
            self._stores_by_name.clear()
            self._stores_by_number.clear()
            self._load_data()

            logger.info(f"Saved {len(stores)} stores for org '{self.org_id}'")
            return True
        except Exception as e:
            logger.error(f"Error saving stores for org '{self.org_id}': {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded

    @property
    def store_count(self) -> int:
        """Get the number of stores."""
        return len(self._stores)


# Global service instances by org_id
_store_services: Dict[str, StoreKnowledgeService] = {}


def get_store_service(org_id: str = "default") -> StoreKnowledgeService:
    """
    Get the store service for an organization.

    Args:
        org_id: Organization identifier

    Returns:
        StoreKnowledgeService instance
    """
    global _store_services
    if org_id not in _store_services:
        _store_services[org_id] = StoreKnowledgeService(org_id)
    return _store_services[org_id]


def reset_store_service(org_id: str = "default") -> None:
    """Reset the store service for an organization (for testing)."""
    global _store_services
    if org_id in _store_services:
        del _store_services[org_id]
