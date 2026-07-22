# gsp-datoviz

Datoviz v0.4 GPU provider for GSP. Publication remains blocked until an ordinary RC3-compatible
Datoviz artifact passes the recorded native and installed-wheel gates.

The package metadata therefore depends only on `gsp-core` for now. For local development, build the
`gsp-core` and `gsp-datoviz` wheels and set `GSP_DATOVIZ_SOURCE=/absolute/path/to/datoviz` before an
explicit probe or session open. Metadata-only discovery remains side-effect-free and does not need
the source checkout. Do not interpret this development bootstrap as compatibility with RC2 or as a
published RC3 dependency contract.
