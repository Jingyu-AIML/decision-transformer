# Decision Transformer

A clean implementation of [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) (Chen et al., 2021).

## Structure

```
src/
  model.py      # DecisionTransformer architecture
  trainer.py    # training loop
  utils.py      # dataset, data loading helpers
scripts/
  train.py      # training entry point
  evaluate.py   # rollout evaluation
configs/
  hopper_medium.yaml
tests/
  test_model.py
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/
```

## Status

- [x] Model architecture
- [x] Trainer
- [x] Dataset utilities
- [ ] D4RL dataset integration
- [ ] Full training run (Hopper-medium)
- [ ] Evaluation against paper baselines
