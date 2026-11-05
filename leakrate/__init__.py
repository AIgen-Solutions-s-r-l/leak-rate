"""leakrate — measuring what escapes when a local model pseudonymizes text for frontier LLMs.

Modules
-------
schema          dataclasses for documents, entities, mappings (see METHODS.md §2, §4)
data            Italian identifier generators + synthetic benchmark generator
pseudonymizers  common interface + regex / Presidio / GLiNER / LLM adapters
metrics         leak rate with cluster-bootstrap CIs, k-anonymity, grammatical agreement
eval            downstream tasks run through a frontier model, reverse mapping
"""

__version__ = "0.1.0"
