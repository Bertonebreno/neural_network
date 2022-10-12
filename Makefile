run:
	poetry run python main.py

all:
	@poetry run black .
	@poetry run mypy . 
	@poetry run isort .
	@poetry run flake8 .
	@poetry run pytest .


init:
	@poetry init
	@poetry install
	@poetry run mypy --install-types --non-interactive