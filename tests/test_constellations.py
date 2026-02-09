from skyboxgen.constellations import constell_from_ra_dec, load_constellation_table


def test_constellation_table_load():
    table = load_constellation_table("constbnd/conbound.c")
    assert len(table.names) == 88 * 3
    assert len(table.bounds) == 324
    con = constell_from_ra_dec(table, 0.0, 0.0)
    assert isinstance(con, str)
    assert len(con) == 3
