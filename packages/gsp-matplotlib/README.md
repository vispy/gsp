# gsp-matplotlib

Matplotlib reference, conformance, and publication provider for GSP.

Install it alongside the exactly matching `gsp-core` version. The provider imports as
`gsp_matplotlib` and registers the `matplotlib` backend lazily with GSP.

Matplotlib is the deterministic reference and publication backend. GPU-oriented depth, raster, and
billboard behavior may use explicitly documented adaptations rather than claiming pixel identity
with GPU providers.
