import os
import torch
import marl_gang_train as trainer

original = trainer.ppo_update_batch


def capture(agents, batch):
    path = os.environ['HELIOS_CAPTURE_PATH']
    torch.save(dict(agents=agents, batch=batch), path)
    print('Saved exact pre-update agents/optimizers and batch:', path, flush=True)
    return original(agents, batch)


trainer.ppo_update_batch = capture
trainer.train()
