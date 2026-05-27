"""Smoke tests to verify the package scaffold is importable."""


def test_package_import():
    import cricket_analytics

    assert hasattr(cricket_analytics, "__version__")
    assert cricket_analytics.__version__ == "0.1.0"


def test_submodules_import():
    import cricket_analytics.data
    import cricket_analytics.models
    import cricket_analytics.scraper
    import cricket_analytics.visualisation

    assert cricket_analytics.data is not None
    assert cricket_analytics.models is not None
    assert cricket_analytics.scraper is not None
    assert cricket_analytics.visualisation is not None


def test_cli_import():
    from cricket_analytics.cli import app

    assert app is not None
