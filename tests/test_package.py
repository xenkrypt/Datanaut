import datanaut


def test_package_importable_with_version():
    assert isinstance(datanaut.__version__, str)
    assert datanaut.__version__.count(".") >= 2
