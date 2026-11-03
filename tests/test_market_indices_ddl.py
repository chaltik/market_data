"""Guard the index bootstrap from accidental destructive rewrites."""

import re
from pathlib import Path


def test_market_indices_bootstrap_is_non_destructive():
    sql = (Path(__file__).parents[1] / "create_market_indices_tables.sql").read_text()
    statements = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    assert not re.search(r"\bDROP\s+(TABLE|VIEW|SCHEMA|TYPE|INDEX)\b", statements, re.I)
    assert len(re.findall(r"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\b", statements, re.I)) == 5
    assert len(re.findall(r"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\b", statements, re.I)) == 6
    assert statements.rfind("COMMIT;") > statements.rfind("CREATE OR REPLACE FUNCTION")
