install:
		poetry install

lh-class:
		poetry run lh-class

build:
		poetry build

publish:
		poetry publish --dry-run

package-install:
		python3 -m pip install --user dist/*.whl

package-force-reinstall:
		python3 -m pip install --user --force-reinstall dist/*.whl