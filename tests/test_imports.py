# tests/test_imports.py

def test_import_package():
    # Top-level package
    import cnn_aspp

    # Core subpackages/modules that should be importable
    import cnn_aspp.cli
    import cnn_aspp.data
    import cnn_aspp.models
    import cnn_aspp.tasks
    import cnn_aspp.xai

    # A couple of concrete modules
    import cnn_aspp.models.aspp_tiny
    import cnn_aspp.models.plain_cnn
    import cnn_aspp.data.ndws_dataset
