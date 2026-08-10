"""Model Registry access layer.

Nothing is re-exported here on purpose. ``backend.app.registry.factory`` reads
the settings, and ``backend.app.core.config`` reads the registry layout, so a
package that imported its own submodules would close that loop at import time.
Consumers import the submodule they need.
"""
