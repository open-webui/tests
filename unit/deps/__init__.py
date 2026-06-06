"""Per-dependency API-surface + usage contract tests.

One test module per third-party dependency, asserting that the symbols
and behaviours Open WebUI relies on from that package still exist. Run
inside an environment with a bumped version installed, these catch the
"new release removed/renamed/changed an API we use" class of breakage.
"""
