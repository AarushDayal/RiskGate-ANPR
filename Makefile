.PHONY: check test smoke

check: test smoke

test:
	./venv/bin/python -m pytest tests -q

smoke:
	./venv/bin/python -m scripts.run_eval --images 10
