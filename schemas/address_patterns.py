"""
Pydantic models for address pattern analysis responses.

These models ensure structured and validated JSON responses from the AddressPatternAgent
for seamless integration with GIS geocoding operations.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional
from enum import Enum


class PatternType(str, Enum):
    """Enumeration of address field pattern types."""
    COMPOUND_OBJECT = "compound_object"
    INDIVIDUAL_FIELDS = "individual_fields"

class AddressComponent(BaseModel):
    """Access paths for individual address components."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None


class AddressPattern(BaseModel):
    """Detailed pattern information for a specific address type (billing, shipping, etc.)."""
    pattern_type: PatternType
    access_path: Optional[str] = Field(None, description="Base path for compound objects (e.g., 'BillingAddress')")
    access_paths: AddressComponent = Field(description="Specific access paths for each address component")
    data_completeness: float = Field(ge=0.0, le=1.0, description="Percentage of records with complete address data")
    sample_addresses_found: int = Field(ge=0, description="Number of valid addresses found in sample")


class CoordinateFieldNames(BaseModel):
    """Field names for latitude and longitude coordinates."""
    latitude: str = Field(default="latitude", description="Field name for latitude values")
    longitude: str = Field(default="longitude", description="Field name for longitude values")


class GeocodingRecommendations(BaseModel):
    """Recommendations for optimal geocoding strategy."""
    primary_address_type: str = Field(description="Recommended primary address type (billing, shipping, etc.)")
    geocoding_strategy: str = Field(description="Recommended geocoding approach")
    expected_geocoding_success_rate: float = Field(ge=0.0, le=1.0, description="Estimated geocoding success rate")
    coordinate_field_names: CoordinateFieldNames = Field(description="Recommended field names for coordinates")


class AnalysisMetadata(BaseModel):
    """Metadata about the address pattern analysis."""
    total_records_analyzed: int = Field(ge=0, description="Total number of records analyzed")
    unique_field_patterns_found: int = Field(ge=0, description="Number of distinct address patterns identified")
    analysis_confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of the analysis")


class AddressPatternAnalysis(BaseModel):
    """Complete address pattern analysis response with validation."""
    address_patterns: Dict[str, AddressPattern] = Field(description="Identified address patterns by type")
    recommendations: GeocodingRecommendations = Field(description="Geocoding strategy recommendations")
    metadata: AnalysisMetadata = Field(description="Analysis metadata and statistics")

    class Config:
        """Pydantic configuration with example schema."""
        json_schema_extra = {
            "example": {
                "address_patterns": {
                    "billing": {
                        "pattern_type": "compound_object",
                        "access_path": "BillingAddress",
                        "access_paths": {
                            "street": "BillingAddress.street",
                            "city": "BillingAddress.city",
                            "state": "BillingAddress.state",
                            "postalCode": "BillingAddress.postalCode",
                            "country": "BillingAddress.country"
                        },
                        "data_completeness": 0.95,
                        "sample_addresses_found": 3
                    },
                    "shipping": {
                        "pattern_type": "individual_fields",
                        "access_paths": {
                            "street": "ShippingStreet",
                            "city": "ShippingCity",
                            "state": "ShippingState",
                            "postalCode": "ShippingPostalCode",
                            "country": "ShippingCountry"
                        },
                        "data_completeness": 0.65,
                        "sample_addresses_found": 2
                    }
                },
                "recommendations": {
                    "primary_address_type": "billing",
                    "geocoding_strategy": "compound_preferred_with_individual_fallback",
                    "expected_geocoding_success_rate": 0.85,
                    "coordinate_field_names": {
                        "latitude": "latitude",
                        "longitude": "longitude"
                    }
                },
                "metadata": {
                    "total_records_analyzed": 5,
                    "unique_field_patterns_found": 2,
                    "analysis_confidence": 0.95
                }
            }
        }


# Validation helper functions
def validate_address_pattern_analysis(data: dict) -> AddressPatternAnalysis:
    """
    Validate and parse address pattern analysis data.

    Args:
        data: Dictionary containing address pattern analysis data

    Returns:
        Validated AddressPatternAnalysis object

    Raises:
        ValidationError: If data doesn't match the expected schema
    """
    return AddressPatternAnalysis(**data)


def get_address_pattern_schema() -> dict:
    """
    Get the JSON schema for AddressPatternAnalysis.

    Returns:
        JSON schema dictionary for validation and documentation
    """
    return AddressPatternAnalysis.model_json_schema()

