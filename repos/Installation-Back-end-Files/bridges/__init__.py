"""Backend Bridges — Import adapters for Metadron Capital engine.

These bridges connect the backend installations to the engine.
The engine imports from here; backends handle the heavy lifting.

Usage from Metadron-Capital:
    import sys
    sys.path.insert(0, "/path/to/Installation-Back-end-Files")
    from bridges import openbb_bridge, mirofish_bridge, newton_bridge, qlib_bridge, bert_bridge
"""
