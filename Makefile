fine-tune:
	uv run modal run -m src.cua.finetune --config-file-name $(config)

evaluation:
	uv run python -m src.cua.evaluate