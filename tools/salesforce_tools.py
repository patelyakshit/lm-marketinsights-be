import json
import logging
import time
from io import StringIO
from typing import Dict, Any, List, Optional

import pandas as pd
from decouple import config

from utils.azure_storage import upload_to_blob_storage
from utils.error_handlers import SalesforceError, handle_sync_agent_errors
from utils.salesforce_auth import salesforce_auth

logger = logging.getLogger(__name__)
PORT = config("PORT")


class SalesforceToolExecutor:
    """Executes Salesforce operations using simple_salesforce"""

    def __init__(self):
        self.auth_manager = salesforce_auth
        logger.info("SalesforceToolExecutor initialized")

    @handle_sync_agent_errors
    def get_all_objects(self) -> Dict[str, Any]:
        """Tool 1: Retrieve all available Salesforce objects in the organization"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Get all sobjects
            describe_result = sf.describe()
            sobjects = describe_result.get("sobjects", [])

            # Organize objects by type
            objects_by_type = {
                "standard": [],
                "custom": [],
                "total_count": len(sobjects),
            }

            for obj in sobjects:
                # Return only essential info to avoid exceeding WebSocket frame size limits
                # For detailed metadata, use get_all_fields_for_object on specific objects
                obj_info = {
                    "name": obj.get("name"),
                    "label": obj.get("label"),
                }

                if obj.get("custom"):
                    objects_by_type["custom"].append(obj_info)
                else:
                    objects_by_type["standard"].append(obj_info)

            execution_time = time.time() - start_time

            logger.info(
                f"Retrieved {len(sobjects)} Salesforce objects in {execution_time:.2f}s"
            )

            return {
                "success": True,
                "data": objects_by_type,
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error retrieving Salesforce objects: {e}")
            raise SalesforceError(f"Failed to retrieve objects: {str(e)}")

    @handle_sync_agent_errors
    def get_all_fields_for_object(self, object_name: str) -> Dict[str, Any]:
        """Tool 2: Get comprehensive field information for specific objects"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Get object description
            sobject = getattr(sf, object_name)
            describe_result = sobject.describe()

            fields_info = []
            relationships = []

            for field in describe_result.get("fields", []):
                field_info = {
                    "name": field.get("name"),
                    "label": field.get("label"),
                    "type": field.get("type"),
                    "length": field.get("length"),
                    "precision": field.get("precision"),
                    "scale": field.get("scale"),
                    "createable": field.get("createable"),
                    "updateable": field.get("updateable"),
                    "queryable": field.get("nillable"),
                    "required": not field.get("nillable", True),
                    "unique": field.get("unique"),
                    "custom": field.get("custom"),
                    "defaultValue": field.get("defaultValue"),
                    "helpText": field.get("inlineHelpText"),
                }

                # Handle relationship fields
                if field.get("relationshipName"):
                    field_info["relationshipName"] = field.get("relationshipName")
                    field_info["referenceTo"] = field.get("referenceTo", [])
                    relationships.append(
                        {
                            "field": field.get("name"),
                            "relationshipName": field.get("relationshipName"),
                            "referenceTo": field.get("referenceTo", []),
                            "relationshipOrder": field.get("relationshipOrder"),
                        }
                    )

                fields_info.append(field_info)

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "data": {
                    "object_name": object_name,
                    "label": describe_result.get("label"),
                    "fields": fields_info,
                    "relationships": relationships,
                    "total_fields": len(fields_info),
                    "createable": describe_result.get("createable"),
                    "deletable": describe_result.get("deletable"),
                    "queryable": describe_result.get("queryable"),
                    "updateable": describe_result.get("updateable"),
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Retrieved {len(fields_info)} fields for {object_name} in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error retrieving fields for {object_name}: {e}")
            raise SalesforceError(
                f"Failed to retrieve fields for {object_name}: {str(e)}"
            )

    @handle_sync_agent_errors
    def query_soql(self, soql_query: str) -> Dict[str, Any]:
        """Tool 3: Execute SOQL queries against Salesforce"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Execute the SOQL query
            result = sf.query(soql_query)

            records = result.get("records", [])

            # Clean up records (remove attributes)
            cleaned_records = []
            for record in records:
                cleaned_record = {k: v for k, v in record.items() if k != "attributes"}
                cleaned_records.append(cleaned_record)

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "query": soql_query,
                    "records": cleaned_records,
                    "total_size": result.get("totalSize", 0),
                    "done": result.get("done", True),
                    "next_records_url": result.get("nextRecordsUrl"),
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"SOQL query returned {len(cleaned_records)} records in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error executing SOQL query: {e}")
            raise SalesforceError(f"SOQL query failed: {str(e)}", {"query": soql_query})

    @handle_sync_agent_errors
    def query_sosl(self, sosl_query: str) -> Dict[str, Any]:
        """Tool 4: Execute SOSL searches across multiple objects"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Execute the SOSL search
            result = sf.search(sosl_query)

            # SOSL returns a list of records grouped by object type
            search_results = []
            total_records = 0

            for record in result:
                cleaned_record = {k: v for k, v in record.items() if k != "attributes"}
                search_results.append(cleaned_record)
                total_records += 1

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "search_query": sosl_query,
                    "search_records": search_results,
                    "total_found": total_records,
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"SOSL search returned {total_records} records in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error executing SOSL search: {e}")
            raise SalesforceError(
                f"SOSL search failed: {str(e)}", {"query": sosl_query}
            )

    @handle_sync_agent_errors
    def get_record_count(
        self, object_name: str, where_clause: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tool 5: Count records matching criteria without data retrieval"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Build count query
            count_query = f"SELECT COUNT() FROM {object_name}"
            if where_clause:
                count_query += f" WHERE {where_clause}"

            result = sf.query(count_query)
            record_count = result.get("totalSize", 0)

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "object_name": object_name,
                    "where_clause": where_clause,
                    "count": record_count,
                    "query": count_query,
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Count query for {object_name} returned {record_count} records in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error counting records for {object_name}: {e}")
            raise SalesforceError(
                f"Count query failed: {str(e)}",
                {"object": object_name, "where": where_clause},
            )

    @handle_sync_agent_errors
    def query_with_pagination(
        self, soql_query: str, batch_size: int = 2000
    ) -> Dict[str, Any]:
        """Tool 6: Handle large result sets with automatic pagination"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            all_records = []
            query_result = sf.query(soql_query)
            batch_count = 0

            while True:
                batch_count += 1
                records = query_result.get("records", [])

                # Clean records
                for record in records:
                    cleaned_record = {
                        k: v for k, v in record.items() if k != "attributes"
                    }
                    all_records.append(cleaned_record)

                # Check if there are more records
                if query_result.get("done", True):
                    break

                # Get next batch
                next_records_url = query_result.get("nextRecordsUrl")
                if next_records_url:
                    query_result = sf.query_more(next_records_url)
                else:
                    break

                # Safety check to prevent infinite loops
                if len(all_records) >= 50000:  # Reasonable limit
                    logger.warning(f"Pagination stopped at 50,000 records for safety")
                    break

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "query": soql_query,
                    "records": all_records,
                    "total_records": len(all_records),
                    "batches_processed": batch_count,
                    "batch_size": batch_size,
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Paginated query returned {len(all_records)} records in {batch_count} batches, {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error in paginated query: {e}")
            raise SalesforceError(
                f"Paginated query failed: {str(e)}", {"query": soql_query}
            )

    @handle_sync_agent_errors
    def aggregate_query(
        self,
        object_name: str,
        aggregate_fields: List[str],
        group_by_fields: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool 7: Execute aggregate functions (COUNT, SUM, AVG, MIN, MAX)"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Build aggregate query
            select_parts = []
            for field in aggregate_fields:
                select_parts.append(field)

            if group_by_fields:
                select_parts.extend(group_by_fields)

            query = f"SELECT {', '.join(select_parts)} FROM {object_name}"

            if where_clause:
                query += f" WHERE {where_clause}"

            if group_by_fields:
                query += f" GROUP BY {', '.join(group_by_fields)}"

            result = sf.query(query)
            records = result.get("records", [])

            # Clean records
            cleaned_records = []
            for record in records:
                cleaned_record = {k: v for k, v in record.items() if k != "attributes"}
                cleaned_records.append(cleaned_record)

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "object_name": object_name,
                    "aggregate_fields": aggregate_fields,
                    "group_by_fields": group_by_fields,
                    "where_clause": where_clause,
                    "query": query,
                    "results": cleaned_records,
                    "total_results": len(cleaned_records),
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Aggregate query returned {len(cleaned_records)} results in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error in aggregate query: {e}")
            raise SalesforceError(
                f"Aggregate query failed: {str(e)}",
                {"object": object_name, "aggregates": aggregate_fields},
            )

    @handle_sync_agent_errors
    def get_object_relationships(
        self, object_name: str, relationship_depth: int = 2
    ) -> Dict[str, Any]:
        """Tool 8: Map parent-child and lookup relationships between objects"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Get object description
            sobject = getattr(sf, object_name)
            describe_result = sobject.describe()

            relationships = {
                "parent_relationships": [],  # Lookup/Master-Detail to other objects
                "child_relationships": [],  # Objects that reference this object
                "object_name": object_name,
            }

            # Get parent relationships (lookup and master-detail fields)
            for field in describe_result.get("fields", []):
                if field.get("type") in ["reference", "id"] and field.get(
                    "referenceTo"
                ):
                    for ref_object in field.get("referenceTo", []):
                        relationship = {
                            "field_name": field.get("name"),
                            "relationship_name": field.get("relationshipName"),
                            "related_object": ref_object,
                            "relationship_type": (
                                "lookup" if field.get("nillable") else "master_detail"
                            ),
                            "createable": field.get("createable"),
                            "updateable": field.get("updateable"),
                        }
                        relationships["parent_relationships"].append(relationship)

            # Get child relationships
            for child_rel in describe_result.get("childRelationships", []):
                if child_rel.get("childSObject"):
                    relationship = {
                        "child_object": child_rel.get("childSObject"),
                        "field_name": child_rel.get("field"),
                        "relationship_name": child_rel.get("relationshipName"),
                        "cascadeDelete": child_rel.get("cascadeDelete"),
                        "deprecatedAndHidden": child_rel.get("deprecatedAndHidden"),
                    }
                    relationships["child_relationships"].append(relationship)

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": relationships,
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            total_relationships = len(relationships["parent_relationships"]) + len(
                relationships["child_relationships"]
            )
            logger.info(
                f"Found {total_relationships} relationships for {object_name} in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error getting relationships for {object_name}: {e}")
            raise SalesforceError(
                f"Failed to get relationships: {str(e)}", {"object": object_name}
            )

    @handle_sync_agent_errors
    async def export_query_to_csv(
        self, soql_query: str, include_headers: bool = True
    ) -> Dict[str, Any]:
        """Tool 9: Export query results directly to CSV format"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Execute query
            result = sf.query(soql_query)
            records = result.get("records", [])

            if not records:
                return {
                    "success": True,
                    "data": {
                        "message": "No records found for export",
                        "query": soql_query,
                        "records_exported": 0,
                    },
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time(),
                }

            # Clean records and convert to DataFrame
            cleaned_records = []
            for record in records:
                cleaned_record = {k: v for k, v in record.items() if k != "attributes"}
                cleaned_records.append(cleaned_record)

            df = pd.DataFrame(cleaned_records)

            timestamp = int(time.time())
            filename = f"salesforce_export_{timestamp}.csv"

            # Convert DataFrame to CSV in memory
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, header=include_headers)
            csv_content = csv_buffer.getvalue()

            # Upload to Azure Blob Storage (await since this function is now async)
            upload_result = await upload_to_blob_storage(
                file_content=csv_content,
                filename=filename,
                content_type="text/csv",
            )

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "query": soql_query,
                    "file_path": upload_result["sas_url"],
                    "filename": filename,
                    "media_url": upload_result["sas_url"],
                    "download_url": upload_result["sas_url"],
                    "records_exported": len(cleaned_records),
                    "columns": list(df.columns),
                    "include_headers": include_headers,
                    "csv_preview": (
                        df.head().to_dict("records") if len(df) > 0 else []
                    ),
                    "storage_type": upload_result["storage_type"],
                    "blob_container": upload_result["container"],
                    "sas_expires_at": upload_result["sas_expires_at"],
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Exported {len(cleaned_records)} records to Azure Blob Storage: {filename} in {execution_time:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise SalesforceError(
                f"CSV export failed: {str(e)}",
                {"query": soql_query},
            )

    @handle_sync_agent_errors
    def validate_soql_syntax(self, soql_query: str) -> Dict[str, Any]:
        """Tool 10: Validate SOQL syntax before execution"""
        start_time = time.time()

        try:
            sf = self.auth_manager.get_connection()

            # Try to execute the query with LIMIT 0 to validate syntax without returning data
            validation_query = f"{soql_query.rstrip()} LIMIT 0"

            try:
                result = sf.query(validation_query)

                # If we get here, the syntax is valid
                validation_result = {
                    "is_valid": True,
                    "syntax_errors": [],
                    "warnings": [],
                    "query": soql_query,
                    "validation_method": "execution_test",
                }

                # Check for potential performance issues
                warnings = []
                query_lower = soql_query.lower()

                if "select *" in query_lower:
                    warnings.append("Avoid using SELECT * - specify only needed fields")

                if "limit" not in query_lower:
                    warnings.append("Consider adding a LIMIT clause for large datasets")

                if query_lower.count("select") > 1:
                    warnings.append(
                        "Complex nested queries detected - verify performance"
                    )

                validation_result["warnings"] = warnings

            except Exception as query_error:
                # Parse the error to provide helpful feedback
                error_message = str(query_error)

                validation_result = {
                    "is_valid": False,
                    "syntax_errors": [error_message],
                    "warnings": [],
                    "query": soql_query,
                    "validation_method": "execution_test",
                }

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": validation_result,
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            status = "valid" if validation_result["is_valid"] else "invalid"
            logger.info(f"SOQL validation completed: {status} in {execution_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error validating SOQL: {e}")
            raise SalesforceError(
                f"SOQL validation failed: {str(e)}", {"query": soql_query}
            )

    def _extract_address_fields(
        self, records: List[Dict]
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Extract address fields from Salesforce records.

        Detects address patterns:
        - Split fields: BillingStreet, BillingCity, BillingState, etc.
        - Compound objects: BillingAddress as nested dict
        - Multiple types: billing, shipping, mailing

        Args:
            records: List of Salesforce records

        Returns:
            {record_index: {label: {address1, address2, city, state, country, postal_code, full}}}
        """
        result = {}

        for idx, record in enumerate(records):
            if not record:
                continue

            addresses = {}
            low_keys = {k.lower(): k for k in record.keys()}

            # Salesforce-style split fields with prefixes (Billing/Shipping/Mailing)
            prefixes = ["billing", "shipping", "mailing", "other"]
            suffix_map = {
                "street": "address1",
                "city": "city",
                "state": "state",
                "statecode": "state",
                "postalcode": "postal_code",
                "country": "country",
                "countrycode": "country",
            }

            for prefix in prefixes:
                components = {}
                found_any = False

                for lk, orig in low_keys.items():
                    if not lk.startswith(prefix):
                        continue

                    for suffix, target in suffix_map.items():
                        if lk == f"{prefix}{suffix}":
                            value = record.get(orig)
                            if value:
                                components[target] = str(value).strip()
                                found_any = True

                if found_any:
                    # Build full address string
                    parts = []
                    if components.get("address1"):
                        parts.append(components["address1"])
                    if components.get("city"):
                        parts.append(components["city"])

                    state_zip = []
                    if components.get("state"):
                        state_zip.append(components["state"])
                    if components.get("postal_code"):
                        state_zip.append(components["postal_code"])
                    if state_zip:
                        parts.append(" ".join(state_zip))

                    if components.get("country"):
                        parts.append(components["country"])

                    components["full"] = ", ".join(parts)

                    if components["full"]:
                        addresses[prefix] = components

            if addresses:
                result[idx] = addresses

        return result

    async def _geocode_addresses_batch(
        self, addresses: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Batch geocode addresses using ArcGIS REST API.

        Args:
            addresses: List of address strings

        Returns:
            List of geocode results: [{latitude, longitude, score, match_address}, None, ...]
        """
        import os
        import aiohttp

        if not addresses:
            return []

        api_key = os.getenv("ARCGIS_GEOLOCATION_API_KEY", "")
        if not api_key:
            logger.warning("ARCGIS_GEOLOCATION_API_KEY not set - geocoding will fail")
            return [None] * len(addresses)

        geocode_url = "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"

        results = []
        CHUNK_SIZE = 50  # Process 50 addresses at a time

        for i in range(0, len(addresses), CHUNK_SIZE):
            chunk = addresses[i : i + CHUNK_SIZE]

            # Geocode each address in the chunk
            for address in chunk:
                params = {
                    "singleLine": address,
                    "f": "json",
                    "token": api_key,
                    "maxLocations": 1,
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(geocode_url, params=params) as response:
                            if response.status != 200:
                                logger.warning(
                                    f"Geocoding failed for '{address}': HTTP {response.status}"
                                )
                                results.append(None)
                                continue

                            data = await response.json()
                            candidates = data.get("candidates", [])

                            if not candidates:
                                logger.info(f"No geocode results for '{address}'")
                                results.append(None)
                                continue

                            # Get best match
                            best = candidates[0]
                            location = best.get("location", {})

                            results.append(
                                {
                                    "latitude": location.get("y"),
                                    "longitude": location.get("x"),
                                    "score": best.get("score", 0),
                                    "match_address": best.get("address", address),
                                }
                            )
                except Exception as e:
                    logger.error(f"Error geocoding '{address}': {e}")
                    results.append(None)

        return results

    async def _annotate_records_with_geocodes(self, records: List[Dict]) -> List[Dict]:
        """
        Detect address fields and geocode them, adding _geocodes to each record.

        Args:
            records: List of Salesforce records

        Returns:
            List of records with _geocodes key added
        """
        if not records:
            return []

        # Extract addresses
        address_map = self._extract_address_fields(records)

        if not address_map:
            logger.info("No address fields found in records")
            return records

        # Flatten addresses for batch geocoding
        address_list = []
        address_metadata = []  # [(record_idx, label), ...]

        for record_idx, address_dict in address_map.items():
            for label, components in address_dict.items():
                full_address = components.get("full")
                if full_address:
                    address_list.append(full_address)
                    address_metadata.append((record_idx, label))

        # Batch geocode
        geocode_results = await self._geocode_addresses_batch(address_list)

        # Map results back to records
        annotated_records = []
        for idx, record in enumerate(records):
            new_record = {**record}

            if idx in address_map:
                geocodes = {}

                # Find geocode results for this record
                for i, (rec_idx, label) in enumerate(address_metadata):
                    if rec_idx == idx and i < len(geocode_results):
                        geocodes[label] = geocode_results[i]

                if geocodes:
                    new_record["_geocodes"] = geocodes

            annotated_records.append(new_record)

        return annotated_records

    def _records_to_geojson(
        self, records: List[Dict], geocodes_col: str = "_geocodes"
    ) -> Dict[str, Any]:
        """
        Convert geocoded records to GeoJSON FeatureCollection.

        Args:
            records: List of records with _geocodes key
            geocodes_col: Column name containing geocode data

        Returns:
            GeoJSON FeatureCollection dict
        """
        features = []

        for record in records:
            geocodes = record.get(geocodes_col)
            if not geocodes or not isinstance(geocodes, dict):
                continue

            # Create one Feature per address label
            for label, geocode in geocodes.items():
                if not geocode:
                    continue

                lat = geocode.get("latitude")
                lon = geocode.get("longitude")

                if lat is None or lon is None:
                    continue

                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except (ValueError, TypeError):
                    continue

                # Build properties (exclude _geocodes column)
                properties = {k: v for k, v in record.items() if k != geocodes_col}
                properties["label"] = label
                properties["geocode"] = {
                    "latitude": lat_f,
                    "longitude": lon_f,
                    "score": geocode.get("score", 0),
                    "match_address": geocode.get("match_address", ""),
                }

                # Create Feature
                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]},
                    "properties": properties,
                }

                # Add ID if available
                if "Id" in record:
                    feature["id"] = record["Id"]

                features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    @handle_sync_agent_errors
    async def export_query_to_geojson(
        self, soql_query: str, use_pagination: bool = False, max_records: int = 2000
    ) -> Dict[str, Any]:
        """
        Execute SOQL query, geocode addresses, and return GeoJSON FeatureCollection.

        Args:
            soql_query: SOQL query (should include address fields)
            use_pagination: If True, fetch all records using pagination (up to max_records)
            max_records: Maximum records to fetch when use_pagination=True (default 2000, max 50000)

        Returns:
            {
                "success": True,
                "data": {
                    "query": "...",
                    "geojson": {...},
                    "record_count": N,
                    "geocoded_count": M,
                    "address_labels_found": [...],
                    "pagination_used": True/False
                },
                "execution_time": X,
                "timestamp": T
            }
        """
        start_time = time.time()

        try:
            # Choose query method based on pagination flag
            if use_pagination:
                # Use pagination for large datasets
                query_result = self.query_with_pagination(soql_query)
            else:
                # Use regular query (respects LIMIT in query)
                query_result = self.query_soql(soql_query)

            if not query_result.get("success"):
                raise SalesforceError("SOQL query failed", query_result)

            records = query_result.get("data", {}).get("records", [])

            # Apply max_records limit if using pagination
            if use_pagination and len(records) > max_records:
                logger.warning(
                    f"Limiting records from {len(records)} to {max_records} for geocoding performance"
                )
                records = records[:max_records]

            if not records:
                return {
                    "success": True,
                    "data": {
                        "query": soql_query,
                        "geojson": {"type": "FeatureCollection", "features": []},
                        "record_count": 0,
                        "geocoded_count": 0,
                        "message": "No records found",
                        "pagination_used": use_pagination,
                    },
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time(),
                }

            # Annotate with geocodes
            annotated_records = await self._annotate_records_with_geocodes(records)

            # Convert to GeoJSON
            geojson = self._records_to_geojson(annotated_records)

            # Collect stats
            geocoded_count = len(geojson["features"])
            address_labels = set()
            for record in annotated_records:
                if "_geocodes" in record:
                    address_labels.update(record["_geocodes"].keys())

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "query": soql_query,
                    "geojson": geojson,
                    "record_count": len(records),
                    "geocoded_count": geocoded_count,
                    "address_labels_found": list(address_labels),
                    "pagination_used": use_pagination,
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            timestamp = int(time.time())
            filename = f"salesforce_export_geojson_{timestamp}.geojson"

            # Convert GeoJSON to string
            geojson_content = json.dumps(geojson)

            # Upload to Azure Blob Storage (await since this function is async)
            upload_result = await upload_to_blob_storage(
                file_content=geojson_content,
                filename=filename,
                content_type="application/geo+json",
            )

            response["file_information"] = {
                "filename": filename,
                "media_url": upload_result["sas_url"],
                "download_url": upload_result["sas_url"],
                "storage_type": upload_result["storage_type"],
                "blob_container": upload_result["container"],
                "sas_expires_at": upload_result["sas_expires_at"],
            }

            logger.info(
                f"Exported {geocoded_count} geocoded features to Azure Blob Storage: {filename} "
                f"{'(paginated)' if use_pagination else '(single batch)'} in {execution_time:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error exporting to GeoJSON: {e}")
            raise SalesforceError(
                f"GeoJSON export failed: {str(e)}", {"query": soql_query}
            )

    @handle_sync_agent_errors
    async def geocode_existing_records(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Geocode already-fetched Salesforce records without re-querying.

        Use this when records are already available from query_soql or query_with_pagination.
        This avoids double querying and improves performance.

        Args:
            records: List of Salesforce records (from query_soql or query_with_pagination response)

        Returns:
            Same format as export_query_to_geojson but without query field
        """
        start_time = time.time()

        try:
            if not records:
                return {
                    "success": False,
                    "data": {
                        "geojson": {"type": "FeatureCollection", "features": []},
                        "record_count": 0,
                        "geocoded_count": 0,
                        "message": "No records provided",
                    },
                    "execution_time": time.time() - start_time,
                    "timestamp": time.time(),
                }

            # Annotate with geocodes
            annotated_records = await self._annotate_records_with_geocodes(records)

            # Convert to GeoJSON
            geojson = self._records_to_geojson(annotated_records)

            # Collect stats
            geocoded_count = len(geojson["features"])
            address_labels = set()
            for record in annotated_records:
                if "_geocodes" in record:
                    address_labels.update(record["_geocodes"].keys())

            execution_time = time.time() - start_time

            response = {
                "success": True,
                "data": {
                    "geojson": geojson,
                    "record_count": len(records),
                    "geocoded_count": geocoded_count,
                    "address_labels_found": list(address_labels),
                },
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            logger.info(
                f"Geocoded {geocoded_count} features from {len(records)} existing records in {execution_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error geocoding existing records: {e}")
            raise SalesforceError(f"Geocoding existing records failed: {str(e)}")


# Global instance
salesforce_tools = SalesforceToolExecutor()
