def test_training_module_imports():
    import laban_rl.training as training

    assert hasattr(training, "train_ppo")
