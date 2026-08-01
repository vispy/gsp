# GSP

GSP is a backend-independent graphics session protocol for scientific visualization. Producers create semantic scene records for a capability-negotiated session, and backend adapters execute the accepted operations.

Use the [protocol and backend guide](protocol-and-backends.md) for installation, session ownership, capability checks, queries, and current backend limitations.

Use the [GSP 0.2 specification](specification/index.md) for the canonical protocol model, conformance language, semantic contracts, and normative registries. The specification is a consolidation draft; accepted detailed topic specifications remain authoritative where requirement-level migration is incomplete.

VisPy2 is the intended high-level plotting producer. Matplotlib is the reference and publication backend, while Datoviz v0.4 is the capability-gated GPU backend.
