# Offline coastline context

`coastlines_50m.json.gz` is a prepared **Natural Earth 1:50 million coastline** layer, version **4.1.0** as recorded in the downloaded archive's `VERSION.txt` (the landing page currently labels an older 4.0.0 release).

- [Dataset description](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-coastline/)
- [Source ZIP](https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip)
- [Natural Earth terms: public domain; modification and redistribution permitted](https://www.naturalearthdata.com/about/terms-of-use/)
- Retrieved: 2026-09-10
- Source ZIP SHA-256: `640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde`

The prepared file contains all 1,429 polyline parts and 60,416 vertices, rounded to five decimal degrees and gzip-compressed to 472,018 bytes. No vertex simplification, land-polygon filling, tile service or runtime map request is used. Longitude/latitude vertices are in degrees. The runtime splits dateline jumps and repeats only the visible segments.

This is geographic context, **not 50 m data or accurate 5 m shoreline evidence**. The scale is 1:50,000,000. Coastlines do not make the plots equal-area or validate the coverage model. No land polygons are filled, so polygon holes are not inferred from open coastline parts.

To regenerate, download the source ZIP separately, then run from the repository root:

```sh
python Tutorial/scripts/prepare_coastlines.py /path/to/ne_50m_coastline.zip
```

The converter uses only the Python standard library and performs no network request. Source metadata and hash are also embedded in the prepared JSON. The notebook resolves this asset relative to `eo_visuals.py`, not the working directory. A missing/corrupt asset produces a labelled grid-only map; it is never fetched automatically.
