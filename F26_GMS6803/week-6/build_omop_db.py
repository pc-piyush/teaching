# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb", "pandas", "numpy"]
# ///
"""Build omop.duckdb: a synthetic OMOP CDM v5.4 subset for the SQL basics deck.

Concept IDs are real standard OMOP concepts; people, dates and values are
simulated. Run with:  uv run build_omop_db.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).with_name("omop.duckdb")

_rng = np.random.default_rng(42)
_N = 1200
_t0 = pd.Timestamp("2016-01-01")
_span = (pd.Timestamp("2024-12-31") - _t0).days

concept_df = pd.DataFrame(
    [
        (8507, "MALE", "Gender", "Gender"),
        (8532, "FEMALE", "Gender", "Gender"),
        (8527, "White", "Race", "Race"),
        (8516, "Black or African American", "Race", "Race"),
        (8515, "Asian", "Race", "Race"),
        (38003563, "Hispanic or Latino", "Ethnicity", "Ethnicity"),
        (38003564, "Not Hispanic or Latino", "Ethnicity", "Ethnicity"),
        (9201, "Inpatient Visit", "Visit", "Visit"),
        (9202, "Outpatient Visit", "Visit", "Visit"),
        (9203, "Emergency Room Visit", "Visit", "Visit"),
        (201826, "Type 2 diabetes mellitus", "Condition", "SNOMED"),
        (320128, "Essential hypertension", "Condition", "SNOMED"),
        (317576, "Coronary arteriosclerosis", "Condition", "SNOMED"),
        (255573, "Chronic obstructive lung disease", "Condition", "SNOMED"),
        (46271022, "Chronic kidney disease", "Condition", "SNOMED"),
        (433736, "Obesity", "Condition", "SNOMED"),
        (4329847, "Myocardial infarction", "Condition", "SNOMED"),
        (440383, "Depressive disorder", "Condition", "SNOMED"),
        (197320, "Acute renal failure syndrome", "Condition", "SNOMED"),
        (1503297, "metformin", "Drug", "RxNorm"),
        (1308216, "lisinopril", "Drug", "RxNorm"),
        (1545958, "atorvastatin", "Drug", "RxNorm"),
        (1112807, "aspirin", "Drug", "RxNorm"),
        (1307046, "metoprolol", "Drug", "RxNorm"),
        (1332418, "amlodipine", "Drug", "RxNorm"),
        (3004410, "Hemoglobin A1c/Hemoglobin.total in Blood", "Measurement", "LOINC"),
        (3016723, "Creatinine [Mass/volume] in Serum or Plasma", "Measurement", "LOINC"),
        (3004249, "Systolic blood pressure", "Measurement", "LOINC"),
    ],
    columns=["concept_id", "concept_name", "domain_id", "vocabulary_id"],
)

# person -------------------------------------------------------------
_person = pd.DataFrame({
    "person_id": np.arange(1, _N + 1),
    "gender_concept_id": _rng.choice([8507, 8532], _N, p=[.48, .52]),
    "year_of_birth": _rng.integers(1935, 2006, _N),
    "month_of_birth": _rng.integers(1, 13, _N),
    "race_concept_id": _rng.choice([8527, 8516, 8515, 0], _N, p=[.6, .2, .08, .12]),
    "ethnicity_concept_id": _rng.choice([38003563, 38003564], _N, p=[.25, .75]),
})
_age = dict(zip(_person.person_id, 2024 - _person.year_of_birth))

# visit_occurrence ---------------------------------------------------
_nv = (_rng.poisson(4, _N) + 1).astype(np.intp)  # np.repeat needs intp (int32 in WASM)
_vp = np.repeat(_person.person_id.values, _nv)
_V = len(_vp)
_vtype = _rng.choice([9201, 9202, 9203], _V, p=[.12, .75, .13])
_vstart = _t0 + pd.to_timedelta(_rng.integers(0, _span, _V), unit="D")
_los = np.where(_vtype == 9201, _rng.integers(1, 11, _V), 0)
_visit = pd.DataFrame({
    "visit_occurrence_id": np.arange(1, _V + 1),
    "person_id": _vp,
    "visit_concept_id": _vtype,
    "visit_start_date": _vstart,
    "visit_end_date": _vstart + pd.to_timedelta(_los, unit="D"),
})
_by_person = {p: g for p, g in _visit.groupby("person_id")}

# condition_occurrence ------------------------------------------------
_base = {201826: .16, 320128: .28, 317576: .09, 255573: .07, 46271022: .06,
         433736: .15, 4329847: .03, 440383: .12, 197320: .04}
_crow = []
for _pid, _g in _by_person.items():
    _scale = 0.35 + _age[_pid] / 55
    for _cid, _p in _base.items():
        if _rng.random() < min(_p * _scale, .85):
            _v = _g.iloc[_rng.integers(0, len(_g))]
            _crow.append((_pid, _cid, _v.visit_start_date, _v.visit_occurrence_id))
_cond = pd.DataFrame(_crow, columns=["person_id", "condition_concept_id",
                                     "condition_start_date", "visit_occurrence_id"])
_cond.insert(0, "condition_occurrence_id", np.arange(1, len(_cond) + 1))
_has = _cond.groupby("person_id").condition_concept_id.apply(set).to_dict()

# drug_exposure -------------------------------------------------------
_drow = []
for _pid, _g in _by_person.items():
    _cs = _has.get(_pid, set())
    if 201826 in _cs and _rng.random() < .72:
        _d0 = _cond[(_cond.person_id == _pid) & (_cond.condition_concept_id == 201826)].condition_start_date.iloc[0]
        _drow.append((_pid, 1503297, _d0 + pd.Timedelta(days=int(_rng.integers(0, 240)))))
    if 320128 in _cs:
        for _dc in (1308216, 1332418):
            if _rng.random() < .45:
                _drow.append((_pid, _dc, _g.visit_start_date.iloc[_rng.integers(0, len(_g))]))
    if _cs & {317576, 4329847}:
        for _dc in (1545958, 1112807, 1307046):
            if _rng.random() < .6:
                _drow.append((_pid, _dc, _g.visit_start_date.iloc[_rng.integers(0, len(_g))]))
_drug = pd.DataFrame(_drow, columns=["person_id", "drug_concept_id", "drug_exposure_start_date"])
_drug["days_supply"] = _rng.choice([30, 90], len(_drug))
_drug["drug_exposure_end_date"] = _drug.drug_exposure_start_date + pd.to_timedelta(_drug.days_supply, unit="D")
_drug.insert(0, "drug_exposure_id", np.arange(1, len(_drug) + 1))

# measurement ---------------------------------------------------------
_mrow = []
for _v in _visit.itertuples():
    _cs = _has.get(_v.person_id, set())
    if _rng.random() < .35:
        _a1c = _rng.normal(7.9, 1.3) if 201826 in _cs else _rng.normal(5.4, .35)
        _mrow.append((_v.person_id, 3004410, _v.visit_start_date, round(max(_a1c, 4.0), 1), "%"))
    if _rng.random() < .6:
        _cr = _rng.lognormal(0, .2) * (1.9 if 197320 in _cs and _rng.random() < .5 else 1.0)
        _cr *= 1.35 if 46271022 in _cs else 1.0
        _mrow.append((_v.person_id, 3016723, _v.visit_start_date, round(_cr, 2), "mg/dL"))
    if _rng.random() < .8:
        _sbp = _rng.normal(141 if 320128 in _cs else 122, 13)
        _mrow.append((_v.person_id, 3004249, _v.visit_start_date, round(_sbp), "mmHg"))
_meas = pd.DataFrame(_mrow, columns=["person_id", "measurement_concept_id",
                                     "measurement_date", "value_as_number", "unit_source_value"])
_meas.insert(0, "measurement_id", np.arange(1, len(_meas) + 1))

# write to omop.duckdb with proper DATE columns -----------------------
DB_PATH.unlink(missing_ok=True)
con = duckdb.connect(str(DB_PATH))
for _name, _df in {"person": _person, "visit_occurrence": _visit,
                   "condition_occurrence": _cond, "drug_exposure": _drug,
                   "measurement": _meas, "concept": concept_df}.items():
    con.register("_tmp", _df)
    _cols = ", ".join(
        f"CAST({c} AS DATE) AS {c}" if c.endswith("_date") else c for c in _df.columns
    )
    con.execute(f"CREATE TABLE {_name} AS SELECT {_cols} FROM _tmp")
    con.unregister("_tmp")
con.close()
print(f"wrote {DB_PATH}")
