import pandas as pd


class NameCanonicalizer:
    """Maps team names to a single canonical form.

    Order of application: former-name table -> always-on aliases ->
    optional configurable sensitive merges (off by default for traceability).
    """

    def __init__(self, former_names: pd.DataFrame, aliases: dict, sensitive_merges: dict):
        self._former = {
            str(row.former): str(row.current)
            for row in former_names.itertuples(index=False)
        }
        self._aliases = {str(k): str(v) for k, v in (aliases or {}).items()}
        merges = sensitive_merges or {"enabled": False, "mappings": {}}
        self._sensitive = (
            {str(k): str(v) for k, v in merges.get("mappings", {}).items()}
            if merges.get("enabled", False)
            else {}
        )

    def canonical(self, name: str) -> str:
        name = str(name)
        name = self._former.get(name, name)
        name = self._aliases.get(name, name)
        name = self._sensitive.get(name, name)
        return name

    def mapping_table(self) -> dict:
        table = dict(self._former)
        table.update(self._aliases)
        table.update(self._sensitive)
        return table
