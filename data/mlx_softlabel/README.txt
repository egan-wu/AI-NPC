Next step:
  1. Switch active model to Haiku (per CLAUDE.md).
  2. Author one in-character deflection per row in
     data/mlx_softlabel/deflection_to_author.jsonl
  3. Merge the authored rows back into train.jsonl with the
     existing token prefix preserved.
  4. Train via:  python scripts/03_train_mlx.py --config configs/training_softlabel.yaml
