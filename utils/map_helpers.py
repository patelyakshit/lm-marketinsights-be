from constants.map_constants import zoom_level

ADDR_TYPE_TO_ZOOM_KEY = {
    # Subaddressing
    "Subaddress": "building",

    # Address points / parcels
    "PointAddress": "building",
    "Parcel": "buildings",

    # Streets
    "StreetAddress": "streets",
    "StreetAddressExt": "streets",
    "StreetInt": "streets",
    "StreetMidBlock": "streets",
    "StreetBetween": "streets",
    "StreetName": "streets",
    "DistanceMarker": "streets",

    # Localities / admin
    "Locality": "city",     
    "PostalLoc": "city",
    "PostalExt": "neighborhood",
    "Postal": "neighborhood",

    # Places & features
    "POI": "building",
    "Feature": "city_block",

    # Coordinates & grids
    "LatLong": "city_block",
    "XY": "city_block",
    "YX": "city_block",
    "MGRS": "city",
    "USNG": "city",
}



def get_zoom_for_addr_type(addr_type: str, default="city") -> int:
    key:str = ADDR_TYPE_TO_ZOOM_KEY.get(addr_type, default)
    zoom_level_val: int = zoom_level[key]
    return zoom_level_val
