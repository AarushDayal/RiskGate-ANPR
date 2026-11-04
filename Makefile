.PHONY: check test smoke

check: test smoke

test:
	./venv/bin/python -m pytest tests -q

smoke:
	./venv/bin/python run_eval.py --images 10
