import typing
import pydantic

HiddenString = typing.Annotated[
    str, {
        'json_schema': {
            "hide": True}}]

HiddenBool = typing.Annotated[
    bool, {
        'json_schema': {
            "hide": True}}]

FlightLineColumnName = typing.Annotated[
    str, {
        "json_schema": {
            "x-reference": "flightline-column-name",
            "description": "The column name to rename"}}]

LayerDataName = typing.Annotated[
    str, {
        "json_schema": {
            "x-reference": "layer-data-name",
            "description": "The data group to rename"}}]

Channel = typing.Annotated[
    int, {
        "json_schema": {
            "x-reference": "channel",
            "description": "Channel to perform filter on"}}]

ChannelAndGate = typing.Annotated[
    dict, {
        "json_schema": {
            "x-reference": "channel-gate",
            "properties": {
                "channel": {
                    "x-reference": "channel",
                    "type": "integer",
                    "description": "Channel to perform filter on"},
                "gate": {
                    "x-reference": "gate",
                    "type": "integer",
                    "description": "Timegate index for filter"}}}}]

ChannelAndGateRange = typing.Annotated[
    dict, {
        "json_schema": {
            "x-reference": "channel-gate-range",
            "properties": {
                "channel": {
                    "x-reference": "channel",
                    "type": "integer",
                    "description": "Channel to perform filter on"},
                "start_gate": {
                    "x-reference": "gate",
                    "type": "integer",
                    "description": "Starting timegate index for filter"},
                "end_gate": {
                    "x-reference": "gate",
                    "type": "integer",
                    "description": "Ending timegate index for filter"}}}}]

# from culling.py

ShapeUrl = typing.Annotated[
    pydantic.AnyUrl, {
        "json_schema": {
            "x-url-media-type": "application/zipped-shapefile"}}]

DistanceDict = typing.Annotated[
    dict, {
        "json_schema": {
            "properties": {
                'Gate_Ch01': {
                    "properties": {
                        'first_gate': {
                            "type": "number"},
                        'last_gate': {"type": "number"}}},
                'Gate_Ch02': {"properties": {'first_gate': {"type": "number"},
                                             'last_gate': {"type": "number"}}}}}}]

InversionModelUrls = typing.Annotated[
    dict, {
        "json_schema": {
            "x-reference": "inversion",
            "properties": {"model": {"type": "string",
                                     "format": "url",
                                     "x-url-media-type": "application/x-geophysics-xyz-model"},
                           "fwd": {"type": "string",
                                   "format": "url",
                                   "x-url-media-type": "application/x-geophysics-xyz-data"},
                           "measured": {"type": "string",
                                        "format": "url",
                                        "x-url-media-type": "application/x-geophysics-xyz-model"}}}}]

# from corrections.py

RasterUrl = typing.Annotated[
    pydantic.AnyUrl, {
        "json_schema": {
            "x-url-media-type": "image/geotiff",
            "minLength": 1}}]

FlightlinesList = typing.Annotated[
    list, {
        "json_schema": {
            "items": {
                "type": "string",
                "x-reference": "flightline"}}}]

MovingAverageFilterDict = typing.Annotated[
    dict, {
        "json_schema": {
            "properties": {
                'Gate_Ch01': {
                    "properties": {
                        'first_gate': {
                            "type": "integer"},
                        'last_gate': {"type": "integer"}}},
                'Gate_Ch02': {
                    "properties": {
                        'first_gate': {
                            "type": "integer"},
                        'last_gate': {
                            "type": "integer"}}}}}}]

FlightType = typing.Literal['Production', 'Tie', 'Test', 'High-altitude'] | str

AvailableFlightType = typing.Annotated[
    FlightType, {
        "json_schema": {
            "type": "string",
            "x-reference": "flight_type"}}]

AvailableFlightTypeList = list[AvailableFlightType]
