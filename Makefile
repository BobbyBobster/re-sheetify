### PACKAGE ACTIONS ###
run_train_basic:
ifeq ($(TRAINING_ENV), local)
	python -c 'from sheets.interface.main import train; train(model_type="basic", year_limit=[0], count_limit=5, epochs=10)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import train; train(model_type="basic")'
endif

run_train_onf:
ifeq ($(TRAINING_ENV), local)
	python -c 'from sheets.interface.main import train; train(model_type="onf", year_limit=[0], count_limit=5, epochs=10)'
endif
ifeq ($(TRAINING_ENV), cloud)
	python -c 'from sheets.interface.main import train; train(model_type="onf")'
endif


precompute_all: precompute_cqt precompute_midi

precompute_cqt:
	python scripts/precompute_cqt.py

precompute_midi:
	python scripts/precompute_midi.py


### TESTS ###
test_all:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

test_gcp_setup:
	@pytest \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/test_gcp_setup.py::TestGcpSetup::test_code_get_project


### DATA SOURCES ACTIONS ###


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
	@rm -rf .sheetify_cache
