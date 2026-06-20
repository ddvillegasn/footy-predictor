import pandas as pd

from footy.data.names import NameCanonicalizer


def _former():
    return pd.DataFrame(
        {
            "current": ["Benin"],
            "former": ["Dahomey"],
            "start_date": pd.to_datetime(["1959-11-08"]),
            "end_date": pd.to_datetime(["1975-11-30"]),
        }
    )


def test_former_name_maps_to_current():
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges={"enabled": False, "mappings": {}})
    assert canon.canonical("Dahomey") == "Benin"
    assert canon.canonical("Benin") == "Benin"


def test_alias_applied():
    canon = NameCanonicalizer(
        _former(),
        aliases={"Vietnam Republic": "South Vietnam"},
        sensitive_merges={"enabled": False, "mappings": {}},
    )
    assert canon.canonical("Vietnam Republic") == "South Vietnam"


def test_sensitive_merge_off_by_default():
    merges = {"enabled": False, "mappings": {"West Germany": "Germany"}}
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges=merges)
    assert canon.canonical("West Germany") == "West Germany"


def test_sensitive_merge_on_when_enabled():
    merges = {"enabled": True, "mappings": {"West Germany": "Germany"}}
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges=merges)
    assert canon.canonical("West Germany") == "Germany"


def test_mapping_table_exported():
    canon = NameCanonicalizer(_former(), aliases={}, sensitive_merges={"enabled": False, "mappings": {}})
    table = canon.mapping_table()
    assert table["Dahomey"] == "Benin"
