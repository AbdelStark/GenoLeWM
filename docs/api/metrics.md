# Metrics

GenoLeWM ships a registered metrics registry with a Prometheus
textfile exporter, defined by
[RFC-0013](../rfcs/0013-observability.md) §4. The registry is the
single source of truth: new metrics MUST be added to
`geno_lewm.metrics.METRICS` and the AST linter
([`check_event_names.py`](https://github.com/AbdelStark/GenoLeWM/blob/main/tools/lint/check_event_names.py))
prevents call-site drift.

## Registry

::: geno_lewm.metrics.METRICS
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Primitives

::: geno_lewm.metrics
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - Counter
        - Gauge
        - Histogram
        - HistogramSnapshot
        - MetricSpec
        - MetricKind
        - DEFAULT_HISTOGRAM_BUCKETS_MS
        - DEFAULT_HISTOGRAM_BUCKETS_BYTES

## Accessors

::: geno_lewm.metrics
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - get_counter
        - get_gauge
        - get_histogram
        - snapshot_all

## Prometheus exporter

::: geno_lewm.metrics.export_prometheus_textfile
::: geno_lewm.metrics.metrics_path

Atomicity guarantee: the exporter writes to a `*.tmp` file and renames
on top of the destination, so scrapers always observe either the
previous or the new value, never a torn record.
