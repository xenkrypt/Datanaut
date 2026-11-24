from pathlib import Path

import pandas as pd
import pytest

from datanaut.cli import main
from datanaut.core.validation import RangeRule, RuleSet
from datanaut.core.versioning import Recipe

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "household_energy_sample.csv"


def clean(tmp_path, source=SAMPLE, *extra):
    recipe = tmp_path / "recipe.json"
    code = main(["clean", str(source), "--recipe-out", str(recipe), *extra])
    return code, recipe


# ------------------------------------------------------------------ clean


def test_clean_applies_only_default_selected_fixes_and_writes_outputs(capsys, tmp_path):
    out_csv = tmp_path / "cleaned.csv"
    code, recipe_path = clean(tmp_path, SAMPLE, "--cleaned-out", str(out_csv))
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("[applied]") == 3
    assert "[applied] 6 exact duplicate row(s): keep the first of each. -> V2" in out
    assert "[skipped]" in out and "(not selected by default)" in out
    assert "outlier" in out and "-> V4" in out and "V5" not in out
    assert "<- head" in out and "recipe:" in out and "cleaned data:" in out

    recipe = Recipe.from_json(recipe_path.read_text(encoding="utf-8"))
    assert [s.op.operation for s in recipe.steps] == [
        "drop_duplicates",
        "handle_missing",
        "handle_missing",
    ]
    assert recipe.source["time_columns"] == ["Date", "Time"] and recipe.source["dayfirst"] is True
    assert recipe.source["filename"] == SAMPLE.name

    cleaned = pd.read_csv(out_csv, index_col=0)
    assert len(cleaned) == 2870 and cleaned.index[0] == "2024-01-01 00:00:00"
    assert cleaned["global_active_power"].isna().sum() == 8  # the long dropout is left alone


def test_clean_validates_against_rules_and_reports_failures(capsys, tmp_path):
    rules = tmp_path / "rules.json"
    RuleSet(
        name="plant",
        version="3",
        rules=[
            RangeRule(id="cap", name="Power at most 2 kW", column="global_active_power", max=2),
            RangeRule(id="volt", name="Voltage 200-300", column="voltage", min=200, max=300),
        ],
    ).save_json(rules)
    code, _ = clean(tmp_path, SAMPLE, "--rules", str(rules))
    out = capsys.readouterr().out
    assert code == 0
    assert "Validation against 'plant' v3" in out
    assert "[FAIL] (error) Power at most 2 kW" in out and "[PASS] (error) Voltage 200-300" in out
    assert "some error-severity rules did not pass" in out


def test_bad_rule_file_fails_before_anything_is_written(capsys, tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(
        '{"name": "x", "rules": [{"type": "rnage", "id": "a", "name": "a"}]}', encoding="utf-8"
    )
    code, recipe = clean(tmp_path, SAMPLE, "--rules", str(rules))
    assert code == 1 and not recipe.exists()
    assert "problems" in capsys.readouterr().err


def test_recipe_out_is_required():
    with pytest.raises(SystemExit) as exc:
        main(["clean", str(SAMPLE)])
    assert exc.value.code == 2


def test_ambiguous_dates_stop_clean_without_writing(capsys, tmp_path):
    f = tmp_path / "amb.csv"
    f.write_text("when,v\n01/02/2024,1\n03/04/2024,2\n05/06/2024,3\n", encoding="utf-8")
    code, recipe = clean(tmp_path, f)
    assert code == 2 and not recipe.exists()
    assert "--dayfirst or --monthfirst" in capsys.readouterr().out
    assert clean(tmp_path, f, "--dayfirst")[0] == 0


def test_clean_of_already_clean_data_writes_an_empty_recipe_that_replays(capsys, tmp_path):
    f = tmp_path / "ok.csv"
    f.write_text(
        "ts,v\n"
        + "\n".join(f"2024-01-01 {h:02d}:00:00,{(h * 7) % 11 + 0.5}" for h in range(24))
        + "\n",
        encoding="utf-8",
    )
    code, recipe_path = clean(tmp_path, f)
    assert code == 0 and Recipe.from_json(recipe_path.read_text(encoding="utf-8")).steps == []
    assert main(["replay", str(f), str(recipe_path)]) == 0


def test_formula_like_text_is_neutralized_in_the_cleaned_csv(capsys, tmp_path):
    f = tmp_path / "notes.csv"
    rows = [
        f"2024-01-01 {h:02d}:00:00,{h}.5,{'=HYPERLINK(1)' if h % 3 == 0 else 'fine'}"
        for h in range(12)
    ]
    f.write_text("ts,v,note\n" + "\n".join(rows) + "\n", encoding="utf-8")
    cleaned = tmp_path / "cleaned.csv"
    code, _ = clean(tmp_path, f, "--cleaned-out", str(cleaned))
    assert code == 0
    text = cleaned.read_text(encoding="utf-8")
    assert "'=HYPERLINK(1)" in text
    assert not any(cell.startswith("=") for line in text.splitlines() for cell in line.split(","))
    assert "4 cell(s) starting with =, +, - or @" in capsys.readouterr().out


def test_excel_sheet_is_remembered_in_the_recipe(variants, tmp_path):
    f = tmp_path / "book.xlsx"
    f.write_bytes(variants["xlsx"].data)
    code, recipe_path = clean(tmp_path, f, "--sheet", "data")
    assert code == 0
    assert Recipe.from_json(recipe_path.read_text(encoding="utf-8")).source["sheet"] == "data"
    assert main(["replay", str(f), str(recipe_path)]) == 0


# ------------------------------------------------------------------ replay


def test_replay_verifies_every_fingerprint_and_rebuilds_identical_output(capsys, tmp_path):
    first_csv, second_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    code, recipe = clean(tmp_path, SAMPLE, "--cleaned-out", str(first_csv))
    capsys.readouterr()
    assert code == 0
    assert main(["replay", str(SAMPLE), str(recipe), "--cleaned-out", str(second_csv)]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok]") == 3 and "all 3 step(s) verified" in out
    assert "raw data matches the recipe" in out
    assert first_csv.read_bytes() == second_csv.read_bytes()


def test_replay_on_a_different_file_is_refused(capsys, tmp_path):
    _, recipe = clean(tmp_path)
    capsys.readouterr()
    changed = tmp_path / "changed.csv"
    changed.write_text(
        SAMPLE.read_text(encoding="utf-8").replace("0.672", "0.999", 1), encoding="utf-8"
    )
    assert main(["replay", str(changed), str(recipe)]) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "raw data does not match" in err


def test_replay_reports_the_step_that_no_longer_matches(capsys, tmp_path):
    _, recipe_path = clean(tmp_path)
    capsys.readouterr()
    recipe = Recipe.from_json(recipe_path.read_text(encoding="utf-8"))
    recipe.steps[1] = recipe.steps[1].model_copy(update={"fingerprint": "0" * 64})
    recipe_path.write_text(recipe.to_json(), encoding="utf-8")
    assert main(["replay", str(SAMPLE), str(recipe_path)]) == 1
    assert "Step V3 (handle_missing)" in capsys.readouterr().err


def test_replay_with_missing_or_broken_recipe_is_a_clean_error(capsys, tmp_path):
    assert main(["replay", str(SAMPLE), str(tmp_path / "nope.json")]) == 1
    assert "could not be opened" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["replay", str(SAMPLE), str(bad)]) == 1
    assert "not valid" in capsys.readouterr().err


# ------------------------------------------------------------------ every Phase 1 variant


def test_every_messy_variant_cleans_and_replays_from_the_command_line(variant, capsys, tmp_path):
    f = tmp_path / variant.filename
    f.write_bytes(variant.data)
    out_csv, again_csv = tmp_path / "c.csv", tmp_path / "d.csv"
    code, recipe = clean(tmp_path, f, "--cleaned-out", str(out_csv))
    assert code == 0, capsys.readouterr()
    assert main(["replay", str(f), str(recipe), "--cleaned-out", str(again_csv)]) == 0
    assert out_csv.read_bytes() == again_csv.read_bytes()
    assert f.read_bytes() == variant.data  # the input file is never modified


# ------------------------------------------------------------------ review fixes


def test_rule_and_recipe_files_saved_with_a_bom_still_work(capsys, tmp_path):
    rules = tmp_path / "rules.json"
    RuleSet(
        name="p", rules=[RangeRule(id="c", name="cap", column="global_active_power", max=2)]
    ).save_json(rules)
    rules.write_bytes(b"\xef\xbb\xbf" + rules.read_bytes())
    code, recipe = clean(tmp_path, SAMPLE, "--rules", str(rules))
    assert code == 0 and "Validation against 'p'" in capsys.readouterr().out
    recipe.write_bytes(b"\xef\xbb\xbf" + recipe.read_bytes())
    assert main(["replay", str(SAMPLE), str(recipe)]) == 0


def _cap_rules(tmp_path, cap):
    rules = tmp_path / "rules.json"
    RuleSet(
        name="plant",
        rules=[RangeRule(id="cap", name="Power cap", column="global_active_power", max=cap)],
    ).save_json(rules)
    return rules


def test_strict_turns_a_failed_error_rule_into_exit_code_3(capsys, tmp_path):
    failing = _cap_rules(tmp_path, 2)
    assert clean(tmp_path, SAMPLE, "--rules", str(failing))[0] == 0  # default: report only
    capsys.readouterr()
    code, recipe = clean(tmp_path, SAMPLE, "--rules", str(failing), "--strict")
    assert code == 3 and recipe.exists()  # outputs are still written
    assert "[FAIL] (error) Power cap" in capsys.readouterr().out
    passing = _cap_rules(tmp_path, 1000)
    assert clean(tmp_path, SAMPLE, "--rules", str(passing), "--strict")[0] == 0


def test_strict_without_rules_is_a_clear_error(capsys, tmp_path):
    code, recipe = clean(tmp_path, SAMPLE, "--strict")
    assert code == 1 and not recipe.exists()
    assert "--strict needs a rule file" in capsys.readouterr().err
