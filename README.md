# Overture Maps Ingestion via Spark Declarative Pipelines

A Spark Declarative Pipeline that reads four Overture Maps themes
(`places`, `buildings`, `water`, `division_areas`) from public S3, filters to
continental Europe, converts WKB to native `GEOMETRY`, and lands the result as
Delta tables with Liquid Clustering on the bbox columns for tight per-file
spatial envelopes.

This repository is packaged as a [Databricks Asset Bundle][dab], so it can be
deployed into any workspace by switching the CLI profile passed at deploy
time.

[dab]: https://docs.databricks.com/aws/en/dev-tools/bundles/

## Layout

```
.
├── databricks.yml                       # Bundle definition (no workspace.host)
├── overture_geo_sdp_pipeline.py         # The SDP notebook (unchanged source)
└── resources/
    └── overture_geo_sdp.pipeline.yml    # Pipeline resource definition
```

## Choosing a workspace via `--profile`

`databricks.yml` deliberately does **not** set `workspace.host`. The host is
taken from the `~/.databrickscfg` profile you pass to the CLI, which means a
single command per profile is enough to deploy into a different workspace:

```bash
# Validate against an EMEA demo workspace
databricks bundle validate --profile fevm-geo-sme-emea

# Deploy
databricks bundle deploy --profile fevm-geo-sme-emea

# Trigger a pipeline update
databricks bundle run --profile fevm-geo-sme-emea overture_geo_sdp
```

To deploy into a different workspace, swap the profile name — no edits to
`databricks.yml` required.

## Overriding the catalog / schema

The pipeline writes into `${var.catalog}.${var.schema}`, defaulting to
`geo_sme_emea_catalog.benchmarking`. Override at the CLI:

```bash
databricks bundle deploy --profile <profile> \
  --var "catalog=my_catalog" \
  --var "schema=overture"
```

Other variables you can override the same way:

| Variable              | Default            | Purpose                                      |
| --------------------- | ------------------ | -------------------------------------------- |
| `catalog`             | `geo_sme_emea_catalog` | UC catalog for pipeline output           |
| `schema`              | `benchmarking`     | UC schema within `catalog`                   |
| `pipeline_serverless` | `true`             | Set `false` for classic DBR 18.2+Photon      |
| `pipeline_channel`    | `PREVIEW`          | Required for GEOMETRY preview features       |

## Notes

- The `spark.databricks.delta.geo.preview.{statsWrite,dataSkipping}.enabled`
  keys are set in the pipeline `configuration` so they apply on both
  serverless and classic compute. The duplicate `spark.conf.set(...)` calls in
  the notebook are belt-and-braces for classic mode.
- The pipeline reads from a public Overture S3 bucket
  (`s3://overturemaps-us-west-2`). Make sure the target workspace's UC
  external location / instance profile allows that read.
- `mode: development` is set on the default target, which prefixes the
  deployed pipeline name with `dev_<username>_` to keep shared workspaces
  tidy. Add a second target with `mode: production` if you need a stable
  un-prefixed name.
