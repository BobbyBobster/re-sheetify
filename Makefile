### PACKAGE ACTIONS ###
run_train_basic:
ifeq ($(TRAINING_ENV), local)
	@rm -rf .sheetify_cache
	python -c 'from sheets.interface.main import train; train(model_type="basic", year_limit=[2018], epochs=10)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import train; train(model_type="basic")'
endif

run_train_onf:
ifeq ($(TRAINING_ENV), local)
	@rm -rf .sheetify_cache
	python -c 'from sheets.interface.main import train; train(model_type="onf", year_limit=[2018], epochs=10)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import train; train(model_type="onf")'
endif


run_save_dataset_basic:
ifeq ($(TRAINING_ENV), local)
	python -c 'from sheets.interface.main import save_dataset; save_dataset(model_type="basic", year_limit=[2018], count_limit=30)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import save_dataset; save_dataset(model_type="basic")'
endif

run_save_dataset_onf:
ifeq ($(TRAINING_ENV), local)
	python -c 'from sheets.interface.main import save_dataset; save_dataset(model_type="onf", year_limit=[2018], count_limit=30)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import save_dataset; save_dataset(model_type="onf")'
endif


symlink_and_save_basic: symlink_bucket run_save_dataset_basic




### PRECOMPUTE TRANSFORMATIONS ###
precompute_all: precompute_cqt precompute_midi

precompute_cqt:
	python scripts/precompute_cqt.py

precompute_midi:
	python scripts/precompute_midi.py

precompute_cqt_local:
	python scripts/precompute_cqt.py local

precompute_midi_local:
	python scripts/precompute_midi.py local


### TESTS ###
test_all:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

test_gcp_setup:
	@pytest \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/test_gcp_setup.py::TestGcpSetup::test_code_get_project


### DATA SOURCES ACTIONS ###
# Requires bucket to be mounted at /mnt/gcs
symlink_bucket:
	-rm -rf /code/data
	ln -s /mnt/gcs/data /code/data
	ls -Al /code/data
	ls -AlH /code/data


### CLEANING ###
clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf **/__pycache__ **/*.pyc
	@rm -rf **/build **/dist
	@rm -rf proj-*.dist-info
	@rm -rf proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
	@rm -rf **/.pytest_cache
	@rm -rf **/.mypy_cache
	@rm -rf **/.ruff_cache
	@rm -rf .sheetify_cache
