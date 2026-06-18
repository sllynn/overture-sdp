# Databricks notebook source
# MAGIC %md
# MAGIC # Overture → Delta: Spark Declarative Pipeline
# MAGIC
# MAGIC Defines four pipeline tables — **places**, **buildings**, **water** and
# MAGIC **division_areas** (administrative boundaries) — sourced from Overture
# MAGIC Maps parquet on S3.  All tables:
# MAGIC
# MAGIC * Convert the WKB `geometry` column to the native `GEOMETRY` type so Delta
# MAGIC   writes per-file bounding-box stats (required for `ST_Intersects` file
# MAGIC   skipping).
# MAGIC * Flatten Overture's top-level `bbox` STRUCT into four `DOUBLE` columns
# MAGIC   (`xmin`, `ymin`, `xmax`, `ymax`) that serve as Liquid Clustering keys.
# MAGIC * Use `cluster_by=["xmin","ymin","xmax","ymax"]` (Liquid Clustering) so
# MAGIC   Predictive Optimization continuously co-locates spatially proximate rows
# MAGIC   into the same files, keeping per-file GEOMETRY bounding-box envelopes tight
# MAGIC   and maximising `ST_Intersects` file-skipping effectiveness.
# MAGIC * Carry **table and column descriptions** sourced from the Overture schema
# MAGIC   reference (https://docs.overturemaps.org/schema/reference/).  These are
# MAGIC   declared as an explicit `schema=` on each table so a redeploy elsewhere
# MAGIC   recreates the full documentation automatically.  For columns whose values
# MAGIC   are restricted to a defined set (e.g. building `subtype`/`class`, water
# MAGIC   `subtype`/`class`, place `operating_status`, division `subtype`/`class`)
# MAGIC   the comment spells out the complete "Available values" enumeration.
# MAGIC
# MAGIC ## Why Liquid Clustering rather than Z-order?
# MAGIC * `pipelines.autoOptimize.zOrderCols` targets the old DLT-internal maintenance
# MAGIC   scheduler, which is being retired. Predictive Optimization — the current
# MAGIC   mechanism — explicitly does **not** run ZORDER.
# MAGIC * Liquid Clustering is PO-native: PO understands and continuously maintains
# MAGIC   it without any manual `OPTIMIZE` calls.
# MAGIC * Liquid Clustering is also incremental — only newly-written files need
# MAGIC   re-clustering on each pipeline refresh, unlike full OPTIMIZE ZORDER BY.
# MAGIC
# MAGIC ## Pipeline cluster requirements
# MAGIC * DBR 18.2 + Photon, dedicated / single-user access mode (classic compute).
# MAGIC * Add to **Pipeline Settings → Advanced → Spark config** (or the pipeline
# MAGIC   `configuration` for serverless):
# MAGIC   ```
# MAGIC   spark.databricks.delta.geo.preview.statsWrite.enabled true
# MAGIC   spark.databricks.delta.geo.preview.dataSkipping.enabled true
# MAGIC   ```
# MAGIC * The cluster instance profile / IAM role must have `s3:GetObject` on
# MAGIC   `overturemaps-us-west-2`.
# MAGIC
# MAGIC ## Pipeline target schema
# MAGIC Set **Target schema** in the pipeline configuration to
# MAGIC `geo_sme_emea_catalog.benchmarking` (or any other `catalog.schema`).

# COMMAND ----------

import dlt

# COMMAND ----------

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

OVERTURE_RELEASE = "2026-06-17.0"   # update if a fresher release is preferred
S3_BASE          = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"

# All Overture geometries are WGS-84 (EPSG:4326).
SRID = 4326

# Pre-filter source parquet to continental Europe using Overture's top-level
# `bbox` STRUCT.  These predicates push down into the parquet reader as
# column-chunk min/max comparisons, pruning files at S3 before any geometry
# conversion happens.
_EU = {"xmin": -10.0, "ymin": 35.0, "xmax": 30.0, "ymax": 65.0}
EUROPE_FILTER = (
    f"bbox.xmax >= {_EU['xmin']} AND bbox.xmin <= {_EU['xmax']} AND "
    f"bbox.ymax >= {_EU['ymin']} AND bbox.ymin <= {_EU['ymax']}"
)

# Liquid Clustering keys: the four flattened bbox coordinate columns.
#
# Why these four?
#   - Clustering on (xmin, ymin, xmax, ymax) co-locates features whose
#     bounding boxes overlap the same region into the same Parquet files.
#   - Tighter per-file geometry envelopes mean Delta's native geospatial
#     data-skipping (which consults per-file GEOMETRY min/max stats) can
#     eliminate more files for any given ST_Intersects predicate.
#   - PO reads CLUSTER BY columns natively and continuously rebalances the
#     layout without any manual OPTIMIZE calls.
CLUSTER_BY_COLS = ["xmin", "ymin", "xmax", "ymax"]

# COMMAND ----------

# Belt-and-suspenders: set here in addition to the pipeline Spark config
# (see header).  These calls run during pipeline notebook initialisation,
# before any table function is evaluated.
#
# On serverless compute the Spark-config allowlist may reject these preview
# keys at runtime, which would raise and fail the whole update.  Guard them so
# the pipeline still runs; the authoritative place to set them on serverless is
# the pipeline-level `configuration` (which is applied before the notebook runs).
for _k in (
    "spark.databricks.delta.geo.preview.statsWrite.enabled",
    "spark.databricks.delta.geo.preview.dataSkipping.enabled",
):
    try:
        spark.conf.set(_k, "true")
    except Exception as _e:  # noqa: BLE001 — config may be allowlist-restricted
        print(f"[overture-pipeline] could not set {_k} from notebook: {_e}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Shared reader helper
# ---------------------------------------------------------------------------

def _read_overture(theme: str, ttype: str):
    """
    Read one Overture theme/type from S3, pre-filter to continental Europe,
    convert WKB binary to the native GEOMETRY type, and flatten the bbox
    STRUCT into four top-level DOUBLE columns.

    The flattened columns serve two purposes:
      1. Liquid Clustering keys — PO clusters rows so adjacent files cover
         tight spatial regions, improving ST_Intersects skipping.
      2. Explicit range-predicate columns — callers can combine bbox-range
         filters with ST_Intersects for belt-and-suspenders skipping.
    """
    src = f"{S3_BASE}/theme={theme}/type={ttype}/"
    return (
        spark.read.parquet(src)
        .where(EUROPE_FILTER)
        .selectExpr(
            "* EXCEPT (geometry, bbox)",
            f"ST_GeomFromWKB(geometry, {SRID}) AS geometry",
            "CAST(bbox.xmin AS DOUBLE) AS xmin",
            "CAST(bbox.ymin AS DOUBLE) AS ymin",
            "CAST(bbox.xmax AS DOUBLE) AS xmax",
            "CAST(bbox.ymax AS DOUBLE) AS ymax",
        )
    )

# COMMAND ----------

# ---------------------------------------------------------------------------
# Schema + column-comment definitions
# ---------------------------------------------------------------------------
#
# Each table declares an explicit output schema as an ordered list of
# (column_name, ddl_type, comment) tuples.  `_schema_ddl()` renders these into
# a SQL DDL string that is passed as `schema=` to `@dlt.table`, so Delta records
# the column comments at table-creation time and a redeploy reproduces them.
#
# The DDL types below mirror exactly what the reader emits (verified against the
# live tables), so the explicit schema is a faithful contract rather than a
# transformation.  `geometry` is the native GEOMETRY type at SRID 4326; the four
# bbox columns are DOUBLE.

def _schema_ddl(cols):
    """Render [(name, ddl_type, comment), ...] into a `col TYPE COMMENT '...'` DDL string."""
    parts = []
    for name, dtype, comment in cols:
        part = f"`{name}` {dtype}"
        if comment:
            part += " COMMENT '" + comment.replace("'", "''") + "'"
        parts.append(part)
    return ",\n".join(parts)

# Reusable Overture composite types (identical across themes).
_NAMES_T = (
    "struct<primary:string,common:map<string,string>,"
    "rules:array<struct<variant:string,language:string,"
    "perspectives:struct<mode:string,countries:array<string>>,"
    "value:string,between:array<double>,side:string>>>"
)
_SOURCES_T = (
    "array<struct<property:string,dataset:string,license:string,"
    "record_id:string,update_time:string,confidence:double,between:array<double>>>"
)
_GEOM_T = f"geometry({SRID})"

# Reusable comments shared by several tables.
_ID_C = (
    "A feature ID. This may be an ID associated with the Global Entity Reference "
    "System (GERS) if—and-only-if the feature represents an entity that is part of GERS."
)
_SOURCES_C = "Information about the source data used to assemble the feature."
_NAMES_C = (
    "Multilingual names container.\n"
    "The sub-field `primary` is the most commonly used name."
)
_VERSION_C = "Tracks the version of the record"
_XMIN_C = "Defines the minimum x-coordinate of the bounding box"
_YMIN_C = "Defines the minimum y-coordinate of the bounding box"
_XMAX_C = "Specifies the maximum x-coordinate of the bounding box"
_YMAX_C = "Specifies the maximum y-coordinate of the bounding box"


def _bbox_cols():
    """The four flattened bbox coordinate columns, common to every table."""
    return [
        ("xmin", "double", _XMIN_C),
        ("ymin", "double", _YMIN_C),
        ("xmax", "double", _XMAX_C),
        ("ymax", "double", _YMAX_C),
    ]

# COMMAND ----------

# ---------------------------------------------------------------------------
# places — theme=places, type=place
# ---------------------------------------------------------------------------

PLACES_COMMENT = (
    "Overture Maps places — continental Europe subset. "
    "GEOMETRY column carries per-file bbox stats for ST_Intersects file "
    "skipping. Liquid Clustering on (xmin, ymin, xmax, ymax) keeps per-file "
    "bounding-box envelopes tight; maintained continuously by Predictive "
    "Optimization.\n\n"
    "Places are point representations of real-world facilities, businesses, "
    "services, or amenities."
)

PLACES_COLS = [
    ("id", "string", _ID_C),
    ("categories",
     "struct<primary:string,alternate:array<string>>",
     "Categories of the place. `primary` is the main category; `alternate` is an "
     "array of additional categories that also apply. Categories are drawn from "
     "the Overture Places category taxonomy (a large hierarchical set of leaf "
     "categories)."),
    ("confidence", "double",
     "The confidence of the existence of the place. A number between 0 and 1. A "
     "confidence of 0 indicates we are certain the place no longer exists (always "
     "paired with operating_status `permanently_closed`); a confidence of 1 "
     "indicates we are certain it exists."),
    ("websites", "array<string>",
     "The websites of the place. Minimum length 1; all items unique."),
    ("emails", "array<string>",
     "The email addresses of the place. Minimum length 1; all items unique."),
    ("socials", "array<string>",
     "The social media URLs of the place. Minimum length 1; all items unique."),
    ("phones", "array<string>",
     "The phone numbers of the place. Minimum length 1; all items unique."),
    ("brand",
     "struct<wikidata:string,names:" + _NAMES_T + ">",
     "The brand of the place. A location with multiple brands is modeled as "
     "multiple separate places, each with its own brand. Contains a `wikidata` "
     "reference and a multilingual names container."),
    ("addresses",
     "array<struct<freeform:string,locality:string,postcode:string,region:string,country:string>>",
     "The address or addresses of the place (freeform, locality, postcode, "
     "region, country). Minimum length 1."),
    ("names", _NAMES_T, _NAMES_C),
    ("sources", _SOURCES_T, _SOURCES_C),
    ("operating_status", "string",
     "An indication of whether a place is: in continued operation, in a temporary "
     "operating hiatus, or closed permanently.\n\n"
     "Available values: open, temporarily_closed, permanently_closed"),
    ("basic_category", "string",
     "The basic-level category of the place — the primary category mapped to a "
     "simplified, human-preferred level of generality within the category "
     "hierarchy. Empty when `categories.primary` is empty."),
    ("taxonomy",
     "struct<primary:string,hierarchy:array<string>,alternates:array<string>>",
     "Structured category representation. `primary` is the most specific "
     "category; `hierarchy` is the ordered taxonomy path from most general to "
     "most specific; `alternates` are additional applicable categories."),
    ("version", "int", _VERSION_C),
    ("geometry", _GEOM_T,
     "Position of the place. Places are point geometries (GeoJSON Point), "
     "expressed in WGS84 geographic coordinates."),
] + _bbox_cols()


@dlt.table(
    name="places",
    comment=PLACES_COMMENT,
    schema=_schema_ddl(PLACES_COLS),
    cluster_by=CLUSTER_BY_COLS,
)
def places():
    return _read_overture("places", "place")

# COMMAND ----------

# ---------------------------------------------------------------------------
# buildings — theme=buildings, type=building
# ---------------------------------------------------------------------------

BUILDINGS_COMMENT = (
    "Overture Maps buildings — continental Europe subset. "
    "GEOMETRY column carries per-file bbox stats for ST_Intersects file "
    "skipping. Liquid Clustering on (xmin, ymin, xmax, ymax) keeps per-file "
    "bounding-box envelopes tight; maintained continuously by Predictive "
    "Optimization.\n\n"
    "A building's geometry represents the two-dimensional footprint of the "
    "building as viewed from directly above, looking down. Fields such as "
    "`height` and `num_floors` allow the three-dimensional shape to be "
    "approximated. Some buildings, identified by the `has_parts` field, have "
    "associated `BuildingPart` features which can be used to generate a more "
    "representative 3D model of the building."
)

BUILDINGS_COLS = [
    ("id", "string", _ID_C),
    ("names", _NAMES_T, _NAMES_C),
    ("sources", _SOURCES_T, _SOURCES_C),
    ("level", "int", "Z-order of the feature where 0 is visual level"),
    ("height", "double",
     "Height of the building or part in meters.\n\n"
     "This is the distance from the lowest point to the highest point."),
    ("min_height", "double",
     "Altitude above ground where the bottom of the building or building part "
     "starts.\n\nIf present, this value indicates that the lowest part of the "
     "building or building part starts is above ground level."),
    ("is_underground", "boolean",
     "Whether the entire building or part is completely below ground.\n\n"
     "The underground flag is useful for display purposes. Buildings and building "
     "parts that are entirely below ground can be styled differently or omitted "
     "from the rendered image.\n\nThis flag is conceptually different from the "
     "level field, which indicates relative z-ordering and, notably, can be "
     "negative even if the building is entirely above-ground."),
    ("num_floors", "int", "Number of above-ground floors of the building or part."),
    ("num_floors_underground", "int",
     "Number of below-ground floors of the building or part."),
    ("min_floor", "int",
     "Start floor of this building or part.\n\nIf present, this value indicates "
     "that the building or part is \"floating\" and its bottom-most floor is above "
     "ground level, usually because it is part of a larger building in which some "
     "parts do reach down to ground level. An example is a building that has an "
     "entry road or driveway at ground level into an interior courtyard, where "
     "part of the building bridges above the entry road. This property may "
     "sometimes be populated when min_height is missing and in these cases can be "
     "used as a proxy for min_height."),
    ("subtype", "string",
     "A broad classification of the current use and purpose of the building.\n\n"
     "If the current use of the building no longer accords with the original "
     "built purpose, the current use should be specified. For example, a building "
     "built as a train station but later converted into a shopping mall would "
     "have the value \"commercial\" rather than \"transportation\".\n\n"
     "Available values: agricultural, civic, commercial, education, "
     "entertainment, industrial, medical, military, outbuilding, religious, "
     "residential, service, transportation"),
    ("class", "string",
     "A more specific classification of the current use and purpose of the "
     "building.\n\nIf the current use of the building no longer accords with the "
     "original built purpose, the current use should be specified.\n\n"
     "Available values: agricultural, allotment_house, apartments, barn, "
     "beach_hut, boathouse, bridge_structure, bungalow, bunker, cabin, carport, "
     "cathedral, chapel, church, civic, college, commercial, cowshed, detached, "
     "digester, dormitory, dwelling_house, factory, farm, farm_auxiliary, "
     "fire_station, garage, garages, ger, glasshouse, government, grandstand, "
     "greenhouse, guardhouse, hangar, hospital, hotel, house, houseboat, hut, "
     "industrial, kindergarten, kiosk, library, manufacture, military, monastery, "
     "mosque, office, outbuilding, parking, pavilion, post_office, presbytery, "
     "public, religious, residential, retail, roof, school, semi, "
     "semidetached_house, service, shed, shrine, silo, slurry_tank, "
     "sports_centre, sports_hall, stable, stadium, static_caravan, stilt_house, "
     "storage_tank, sty, supermarket, synagogue, temple, terrace, toilets, "
     "train_station, transformer_tower, transportation, trullo, university, "
     "warehouse, wayside_shrine"),
    ("facade_color", "string", "Facade color in #rgb or #rrggbb hex notation"),
    ("facade_material", "string",
     "Outer surface material of the facade\n\n"
     "Available values: brick, cement_block, clay, concrete, glass, metal, "
     "plaster, plastic, stone, timber_framing, wood"),
    ("roof_material", "string",
     "Outer surface material of the roof\n\n"
     "Available values: concrete, copper, eternit, glass, grass, gravel, metal, "
     "plastic, roof_tiles, slate, solar_panels, tar_paper, thatch, wood"),
    ("roof_shape", "string",
     "Shape of the roof\n\n"
     "Available values: dome, flat, gabled, gambrel, half_hipped, hipped, "
     "mansard, onion, pyramidal, round, saltbox, sawtooth, skillion, spherical"),
    ("roof_direction", "double", "Bearing of the roof ridge line in degrees"),
    ("roof_orientation", "string",
     "Orientation of the roof shape relative to the footprint shape\n\n"
     "Available values: across, along"),
    ("roof_color", "string", "The roof color in #rgb or #rrggbb hex notation"),
    ("roof_height", "double",
     "Height of the roof in meters.\n\nThis is the distance from the base of the "
     "roof to its highest point"),
    ("has_parts", "boolean",
     "Whether the building has associated building part features"),
    ("version", "int", _VERSION_C),
    ("geometry", _GEOM_T,
     "The building's footprint or roofprint (if traced from aerial/satellite "
     "imagery). Expressed in WGS84 geographic coordinates."),
] + _bbox_cols()


@dlt.table(
    name="buildings",
    comment=BUILDINGS_COMMENT,
    schema=_schema_ddl(BUILDINGS_COLS),
    cluster_by=CLUSTER_BY_COLS,
)
def buildings():
    return _read_overture("buildings", "building")

# COMMAND ----------

# ---------------------------------------------------------------------------
# water — theme=base, type=water
# ---------------------------------------------------------------------------

WATER_COMMENT = (
    "Overture Maps water (base theme) — continental Europe subset. Covers "
    "oceans, seas, lakes, rivers, canals and other water features. GEOMETRY "
    "column carries per-file bbox stats for ST_Intersects file skipping. Liquid "
    "Clustering on (xmin, ymin, xmax, ymax) keeps per-file bounding-box envelopes "
    "tight; maintained continuously by Predictive Optimization.\n\n"
    "Water features represent ocean and inland water bodies.\n\n"
    "In Overture data releases, water features are sourced from OpenStreetMap. "
    "There are two main categories of water feature: ocean and inland water "
    "bodies.\n\n"
    "## Ocean\n\n"
    "The `subytpe` value `\"ocean\"` indicates an ocean area feature whose "
    "geometry represents the surface area of an ocean or part of an ocean. Ocean "
    "area may be tiled into many small polygons of consistent complexity to "
    "ensure manageable geometry. In Overture data releases, ocean area features "
    "are created from OpenStreetMap coastlines data (`natural=coastline`) using a "
    "QA'd version of the output from the OSMCoastline tool. In aggregate, all the "
    "ocean area features represent the inverse of the land features with subtype "
    "`\"land\"` and class `\"land\"`.\n\n"
    "The names and recommended label position for oceans and seas can be found in "
    "features with the subtype `\"physical\"` and the class `\"ocean\"` or "
    "`\"sea\"`.\n\n"
    "## Inland Water\n\n"
    "Subtypes other than `\"ocean\"` (and `\"physical\"`) represent inland water "
    "bodies. In Overture data releases, these features are sourced from the "
    "OpenStreetMap tag `natural=*` where the tag value indicates a water body, as "
    "well as the tags `natural=water`, `waterway=*`, and `water=*`."
)

WATER_COLS = [
    ("id", "string", _ID_C),
    ("names", _NAMES_T,
     "Contains various names for the water feature, including a primary name "
     "along with common names and multiple language variants, allowing for "
     "multi-lingual applications."),
    ("subtype", "string",
     "The broad classification of water body such as river, ocean or lake.\n\n"
     "Values: canal, human_made, lake, ocean, physical, pond, reservoir, river, "
     "spring, stream, wastewater, water"),
    ("class", "string",
     "Further description of the type of water body.\n\n"
     "Values: basin, bay, blowhole, canal, cape, ditch, dock, drain, fairway, "
     "fish_pass, fishpond, geyser, hot_spring, lagoon, lake, moat, ocean, oxbow, "
     "pond, reflecting_pool, reservoir, river, salt_pond, sea, sewage, shoal, "
     "spring, strait, stream, swimming_pool, tidal_channel, wastewater, water, "
     "water_storage, waterfall"),
    ("sources", _SOURCES_T, _SOURCES_C),
    ("source_tags", "map<string,string>",
     "Key-value pairs imported directly from the source data without change.\n\n"
     "This field provides access to raw OSM entity tags for features sourced from "
     "OpenStreetMap"),
    ("level", "int", "Z-order of the feature where 0 is visual level"),
    ("wikidata", "string",
     "A wikidata ID, as found on https://www.wikidata.org/"),
    ("is_intermittent", "boolean",
     "Whether the water body exists intermittently, not permanently"),
    ("is_salt", "boolean", "Whether the water body contains salt water"),
    ("version", "int", _VERSION_C),
    ("geometry", _GEOM_T,
     "Geometry of the water feature (in geographic coordinates), which may be a "
     "point, line string, polygon, or multi-polygon"),
] + _bbox_cols()


@dlt.table(
    name="water",
    comment=WATER_COMMENT,
    schema=_schema_ddl(WATER_COLS),
    cluster_by=CLUSTER_BY_COLS,
)
def water():
    # Water is a *type* within Overture's `base` theme, not a top-level theme.
    return _read_overture("base", "water")

# COMMAND ----------

# ---------------------------------------------------------------------------
# division_areas — theme=divisions, type=division_area
# ---------------------------------------------------------------------------

DIVISION_AREAS_COMMENT = (
    "Overture Maps administrative areas (divisions theme, division_area type) — "
    "continental Europe subset. GEOMETRY column carries per-file bbox stats for "
    "ST_Intersects file skipping. Liquid Clustering on (xmin, ymin, xmax, ymax) "
    "keeps per-file bounding-box envelopes tight; maintained continuously by "
    "Predictive Optimization. NOTE: admin areas are large polygons, so per-file "
    "bbox envelopes overlap heavily and data-skipping yields less here than for "
    "points/footprints — the table is small, so this is fine.\n\n"
    "Division areas are polygons that represent the land or maritime area covered "
    "by a division."
)

# Source order (theme=divisions/type=division_area), with geometry+bbox moved to
# the end by the reader; these files also carry constant `theme`/`type` columns.
DIVISION_AREAS_COLS = [
    ("id", "string", _ID_C),
    ("country", "string",
     "ISO 3166-1 alpha-2 country code of the division this area belongs to."),
    ("sources", _SOURCES_T, _SOURCES_C),
    ("subtype", "string",
     "Category of the division from a finite, hierarchical, ordered list of "
     "categories (e.g. country, region, locality, etc.), similar to a Who's on "
     "First placetype.\n\n"
     "Available values (most general to most specific): country, dependency, "
     "macroregion, region, macrocounty, county, localadmin, locality, borough, "
     "macrohood, neighborhood, microhood"),
    ("admin_level", "int",
     "Integer representing the division's position in its country's "
     "administrative hierarchy, where lower numbers correspond to higher-level "
     "administrative units (countries are 0, regions 1, further subdivisions "
     "greater than 1). Required when subtype is country, dependency, macroregion, "
     "region, macrocounty or county."),
    ("class", "string",
     "Whether the area is the land-clipped boundary or extends into maritime "
     "waters.\n\n"
     "Available values: land (the area does not extend beyond the coastline), "
     "maritime (the area extends beyond the coastline, in most cases to the "
     "extent of the division's territorial sea, if it has one)."),
    ("names", _NAMES_T, _NAMES_C),
    ("is_land", "boolean",
     "A boolean to indicate whether or not the feature geometry represents the "
     "land-clipped, non-maritime boundary. The geometry can be used for map "
     "rendering, cartographic display, and similar purposes."),
    ("is_territorial", "boolean",
     "A boolean to indicate whether or not the feature geometry represents "
     "Overture's best approximation of this place's maritime boundary. For "
     "coastal places, this would tend to include the water area. The geometry can "
     "be used for data processing, reverse-geocoding, and similar purposes."),
    ("region", "string",
     "ISO 3166-2 principal subdivision code of the division this area belongs to."),
    ("division_id", "string",
     "Division ID of the division this area belongs to."),
    ("version", "int", _VERSION_C),
    ("theme", "string",
     "Overture theme partition this feature belongs to (constant: 'divisions')."),
    ("type", "string",
     "Overture type partition this feature belongs to (constant: 'division_area')."),
    ("geometry", _GEOM_T,
     "The area covered by the division. Polygon or MultiPolygon, expressed in "
     "WGS84 geographic coordinates."),
] + _bbox_cols()


@dlt.table(
    name="division_areas",
    comment=DIVISION_AREAS_COMMENT,
    schema=_schema_ddl(DIVISION_AREAS_COLS),
    cluster_by=CLUSTER_BY_COLS,
)
def division_areas():
    # Administrative areas are the `division_area` type within the `divisions`
    # theme. The shared helper handles the WKB->GEOMETRY conversion and bbox
    # flattening identically to the other themes.
    return _read_overture("divisions", "division_area")
