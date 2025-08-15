ifndef SOURCE_FILES
	export SOURCE_FILES:=src
endif
ifndef TEST_FILES
	export TEST_FILES:=tests
endif

.PHONY: lint format test

format:
	uv run ruff format ${SOURCE_FILES} ${TEST_FILES}
	uv run ruff check ${SOURCE_FILES} ${TEST_FILES} --fix-only --exit-zero
	
lint:
	uv run ruff check ${SOURCE_FILES} ${TEST_FILES}
	uv run mypy ${SOURCE_FILES} ${TEST_FILES}

test:
	uv run pytest ${TEST_FILES}

