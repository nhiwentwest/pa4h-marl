import torch
import pytest
import marl_gang_train as trainer


def test_dataset_revision_rejects_old_checkpoint_before_loading_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer, "QUEUE_RELIEF_ENABLED", False)
    monkeypatch.setenv('HELIOS_DATASET_REVISION','completed-diverse-v1')
    monkeypatch.setenv('HELIOS_DATASET_SHA256','new-data')
    path=tmp_path/'old.pt'
    torch.save(dict(semantic_version='helios-a4-constrained-risk-v2'),path)
    with pytest.raises(ValueError,match='legacy resume forbidden'):
        trainer.load_training_state(path,None,None)
    assert trainer.training_semantics()=='helios-a4-completed-diverse-v3'


def test_same_semantics_with_different_dataset_rejects_resume(tmp_path,monkeypatch):
    monkeypatch.setenv('HELIOS_DATASET_REVISION','completed-diverse-v1')
    monkeypatch.setenv('HELIOS_DATASET_SHA256','old-data')
    descriptor=trainer.relief_credit_metadata()
    monkeypatch.setenv('HELIOS_DATASET_SHA256','new-data')
    path=tmp_path/'wrong-data.pt'
    torch.save(dict(schema_version=2,semantic_version=trainer.training_semantics(),
                    simulator_semantics=trainer.SIMULATOR_SEMANTICS,relief_credit=descriptor),path)
    with pytest.raises(ValueError,match='descriptor differs'):
        trainer.load_training_state(path,None,None)


def test_queue_relief_rejects_completed_only_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer, "QUEUE_RELIEF_ENABLED", True)
    assert trainer.training_semantics()=="helios-queue-relief-v4"
    path=tmp_path/"old_completed.pt"
    torch.save(dict(semantic_version="helios-a4-completed-diverse-v3"),path)
    with pytest.raises(ValueError,match="legacy resume forbidden"):
        trainer.load_training_state(path,None,None)
