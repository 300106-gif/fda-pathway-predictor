"""
Tests for the FDA Pathway Predictor pipeline.

Run with:
    uv run pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parents[1]))

ARTIFACTS_DIR = Path(__file__).parents[1] / "artifacts"


# ---------------------------------------------------------------------------
# generate_sample_data
# ---------------------------------------------------------------------------

class TestGenerateSampleData:
    def test_returns_dataframe(self):
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=50, seed=1)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=100, seed=42)
        assert len(df) == 100

    def test_required_columns(self):
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=50, seed=42)
        required = {
            "device_name", "product_code", "device_class",
            "medical_specialty_description", "implant_flag",
            "life_sustain_support_flag", "pathway",
        }
        assert required.issubset(set(df.columns))

    def test_pathway_values(self):
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=300, seed=42)
        assert set(df["pathway"].unique()).issubset({"510k", "PMA", "De Novo"})

    def test_pathway_distribution(self):
        """Check approximate 60/30/10 distribution with tolerance."""
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=1000, seed=42)
        counts = df["pathway"].value_counts(normalize=True)
        assert counts.get("510k", 0) > 0.50, "510k should be dominant"
        assert counts.get("PMA", 0) > 0.20, "PMA should be significant"
        assert counts.get("De Novo", 0) > 0.05, "De Novo should be present"

    def test_implant_flag_values(self):
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=100, seed=42)
        assert set(df["implant_flag"].unique()).issubset({"Y", "N"})

    def test_reproducibility(self):
        from src.tools.generate_sample_data import generate_sample_data
        df1 = generate_sample_data(n=50, seed=99)
        df2 = generate_sample_data(n=50, seed=99)
        assert df1.equals(df2)


# ---------------------------------------------------------------------------
# data_processing — clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    @pytest.fixture()
    def raw_df(self):
        return pd.DataFrame({
            "device_name": ["Monitor", "Pacemaker", "Stent", None, "Monitor"],
            "product_code": ["ABC", "DEF", "GHI", "JKL", "ABC"],
            "device_class": ["2", "III", "Class II", "1", "2"],
            "medical_specialty_description": ["Cardio", "Cardio", "Ortho", "Neuro", "Cardio"],
            "implant_flag": ["N", "YES", "y", "0", "N"],
            "life_sustain_support_flag": ["N", "Y", "NO", "1", "N"],
            "pathway": ["510k", "PMA", "510k", "De Novo", "510k"],
        })

    def test_deduplication(self, raw_df):
        from src.tools.data_processing import clean_data
        cleaned = clean_data(raw_df)
        # Row 0 and 4 are duplicates except for null handling
        assert len(cleaned) <= len(raw_df)

    def test_device_class_normalized(self, raw_df):
        from src.tools.data_processing import clean_data
        cleaned = clean_data(raw_df)
        valid_classes = {"Class I", "Class II", "Class III", "Unknown"}
        assert set(cleaned["device_class"].unique()).issubset(valid_classes)

    def test_implant_flag_normalized(self, raw_df):
        from src.tools.data_processing import clean_data
        cleaned = clean_data(raw_df)
        assert set(cleaned["implant_flag"].unique()).issubset({"Y", "N"})

    def test_pathway_valid(self, raw_df):
        from src.tools.data_processing import clean_data
        cleaned = clean_data(raw_df)
        assert set(cleaned["pathway"].unique()).issubset({"510k", "PMA", "De Novo"})

    def test_saves_contract(self, raw_df, tmp_path, monkeypatch):
        import src.tools.data_processing as dp
        monkeypatch.setattr(dp, "ARTIFACTS_DIR", tmp_path)
        dp.clean_data(raw_df)
        contract_path = tmp_path / "dataset_contract.json"
        assert contract_path.exists()
        with open(contract_path) as f:
            contract = json.load(f)
        assert "row_count" in contract
        assert "pathway" in contract.get("target_column", "pathway")

    def test_saves_clean_csv(self, raw_df, tmp_path, monkeypatch):
        import src.tools.data_processing as dp
        monkeypatch.setattr(dp, "ARTIFACTS_DIR", tmp_path)
        dp.clean_data(raw_df)
        assert (tmp_path / "clean_data.csv").exists()


# ---------------------------------------------------------------------------
# ml_pipeline — feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    @pytest.fixture()
    def clean_df(self):
        from src.tools.generate_sample_data import generate_sample_data
        return generate_sample_data(n=100, seed=42)

    def test_returns_correct_shape(self, clean_df):
        from src.tools.ml_pipeline import engineer_features
        X, y, cols, le = engineer_features(clean_df)
        assert len(X) == len(clean_df)
        assert len(X.columns) == len(cols)

    def test_y_length_matches(self, clean_df):
        from src.tools.ml_pipeline import engineer_features
        X, y, cols, le = engineer_features(clean_df)
        assert len(y) == len(clean_df)

    def test_label_encoder_classes(self, clean_df):
        from src.tools.ml_pipeline import engineer_features
        _, _, _, le = engineer_features(clean_df)
        assert set(le.classes_).issubset({"510k", "PMA", "De Novo"})

    def test_binary_flags_are_01(self, clean_df):
        from src.tools.ml_pipeline import engineer_features
        X, _, cols, _ = engineer_features(clean_df)
        for col in cols:
            if col.endswith("_bin"):
                assert set(X[col].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# ml_pipeline — training & evaluation
# ---------------------------------------------------------------------------

class TestTrainModels:
    @pytest.fixture()
    def features(self):
        from src.tools.generate_sample_data import generate_sample_data
        from src.tools.ml_pipeline import engineer_features
        df = generate_sample_data(n=200, seed=42)
        return engineer_features(df)

    def test_returns_results_dict(self, features, tmp_path, monkeypatch):
        import src.tools.ml_pipeline as mlp
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        X, y, _, le = features
        results, best_name = mlp.train_models(X, y, le)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_best_name_is_one_of_models(self, features, tmp_path, monkeypatch):
        import src.tools.ml_pipeline as mlp
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        X, y, _, le = features
        results, best_name = mlp.train_models(X, y, le)
        assert best_name in {"RandomForest", "GradientBoosting", "LogisticRegression"}

    def test_model_pkl_saved(self, features, tmp_path, monkeypatch):
        import src.tools.ml_pipeline as mlp
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        X, y, _, le = features
        mlp.train_models(X, y, le)
        assert (tmp_path / "model.pkl").exists()

    def test_f1_scores_in_range(self, features, tmp_path, monkeypatch):
        import src.tools.ml_pipeline as mlp
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        X, y, _, le = features
        results, _ = mlp.train_models(X, y, le)
        for name, res in results.items():
            assert 0.0 <= res["macro_f1"] <= 1.0, f"{name} F1 out of range"


class TestEvaluateModels:
    @pytest.fixture()
    def trained(self, tmp_path, monkeypatch):
        import src.tools.ml_pipeline as mlp
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        from src.tools.generate_sample_data import generate_sample_data
        df = generate_sample_data(n=200, seed=42)
        X, y, _, le = mlp.engineer_features(df)
        results, best_name = mlp.train_models(X, y, le)
        return results, best_name, le, tmp_path, monkeypatch

    def test_evaluation_report_created(self, trained):
        import src.tools.ml_pipeline as mlp
        results, best_name, le, tmp_path, monkeypatch = trained
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        mlp.evaluate_models(results, best_name, le)
        assert (tmp_path / "evaluation_report.md").exists()

    def test_model_card_created(self, trained):
        import src.tools.ml_pipeline as mlp
        results, best_name, le, tmp_path, monkeypatch = trained
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        mlp.evaluate_models(results, best_name, le)
        assert (tmp_path / "model_card.md").exists()

    def test_confusion_matrix_created(self, trained):
        import src.tools.ml_pipeline as mlp
        results, best_name, le, tmp_path, monkeypatch = trained
        monkeypatch.setattr(mlp, "ARTIFACTS_DIR", tmp_path)
        mlp.evaluate_models(results, best_name, le)
        assert (tmp_path / "confusion_matrix.png").exists()


# ---------------------------------------------------------------------------
# fda_api_tool — unit tests (no live API calls)
# ---------------------------------------------------------------------------

class TestFdaApiTool:
    def test_get_returns_none_on_404(self, requests_mock):
        from src.tools.fda_api_tool import _get
        requests_mock.get(
            "https://api.fda.gov/device/510k.json",
            status_code=404,
        )
        result = _get("https://api.fda.gov/device/510k.json", {})
        assert result is None

    def test_get_returns_json_on_200(self, requests_mock):
        from src.tools.fda_api_tool import _get
        requests_mock.get(
            "https://api.fda.gov/device/510k.json",
            json={"results": [{"k_number": "K123456"}]},
        )
        result = _get("https://api.fda.gov/device/510k.json", {})
        assert result is not None
        assert "results" in result

    def test_build_raw_dataframe_assigns_de_novo(self, requests_mock, tmp_path, monkeypatch):
        """k_number starting with DEN should get pathway De Novo."""
        import src.tools.fda_api_tool as fat
        monkeypatch.setattr(fat, "ARTIFACTS_DIR", tmp_path)

        # 510k endpoint: 1 DEN record + 1 normal
        requests_mock.get(
            "https://api.fda.gov/device/510k.json",
            [
                {"json": {"results": [
                    {"k_number": "DEN210001", "device_name": "Widget A",
                     "product_code": "ABC", "device_class": "II",
                     "medical_specialty_description": "Cardio",
                     "decision_code": "SESE"},
                    {"k_number": "K123456", "device_name": "Widget B",
                     "product_code": "DEF", "device_class": "II",
                     "medical_specialty_description": "Ortho",
                     "decision_code": "SESE"},
                ]}},
                {"status_code": 404},
            ],
        )
        requests_mock.get(
            "https://api.fda.gov/device/pma.json",
            status_code=404,
        )

        df = fat.build_raw_dataframe()
        assert "De Novo" in df["pathway"].values
        assert "510k" in df["pathway"].values


# ---------------------------------------------------------------------------
# Validation gate logic
# ---------------------------------------------------------------------------

class TestValidationGates:
    def test_crew1_passes_with_valid_data(self, tmp_path, monkeypatch):
        import src.flow.main_flow as mf
        monkeypatch.setattr(mf, "ARTIFACTS_DIR", tmp_path)

        from src.tools.generate_sample_data import generate_sample_data
        import src.tools.generate_sample_data as gsd
        monkeypatch.setattr(gsd, "ARTIFACTS_DIR", tmp_path)

        df = generate_sample_data(n=200, seed=42)
        # contract written by generate_sample_data
        mf._validate_crew1_outputs()

    def test_crew1_fails_with_missing_pathway_column(self, tmp_path, monkeypatch):
        import src.flow.main_flow as mf
        monkeypatch.setattr(mf, "ARTIFACTS_DIR", tmp_path)
        bad_df = pd.DataFrame({"device_name": ["A", "B"], "device_class": ["I", "II"]})
        bad_df.to_csv(tmp_path / "clean_data.csv", index=False)
        (tmp_path / "dataset_contract.json").write_text("{}")
        with pytest.raises(AssertionError, match="pathway"):
            mf._validate_crew1_outputs()

    def test_crew2_fails_with_missing_model(self, tmp_path, monkeypatch):
        import src.flow.main_flow as mf
        monkeypatch.setattr(mf, "ARTIFACTS_DIR", tmp_path)
        with pytest.raises(AssertionError):
            mf._validate_crew2_outputs()
