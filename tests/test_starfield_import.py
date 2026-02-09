from skyboxgen.starfield import load_starfield_systems, write_game_stars_csv


def test_starfield_import_with_xyz_and_fictional(tmp_path):
    content = (
        "Star_system,Faction,Bodies,Inorg_Res,Org_Res,Flora,Fauna,Level,x,y,z,Fictional\n"
        "Kryx\t\t20\t19\t8\tYes\tYes\t35\t-1.0\t2.0\t3.0\ttrue\n"
        "Alpha Centauri\t\t\t\t\t\t\t\t0.1\t0.2\t0.3\tfalse\n"
    )
    src = tmp_path / "starfield.txt"
    src.write_text(content, encoding="utf-8")
    systems = load_starfield_systems(str(src))
    assert systems[0].name == "Kryx"
    assert systems[0].fictional is True
    out = tmp_path / "game_stars.csv"
    write_game_stars_csv(systems, str(out))
    data = out.read_text(encoding="utf-8")
    assert "Kryx" in data
    assert "true" in data
