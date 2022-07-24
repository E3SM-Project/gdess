import pytest

from gdess.data_source.multiset import Multiset


def test_multiset_prints():
    ms = Multiset()
    assert len(str(ms)) > 0
