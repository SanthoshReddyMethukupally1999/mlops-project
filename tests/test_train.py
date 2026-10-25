from src.train import train


def test_model_trains_and_saves(tmp_path):
    model_file = tmp_path / "model.joblib"
    accuracy = train(model_path=model_file)

    assert accuracy > 0.9
    assert model_file.exists()
