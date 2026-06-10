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


### TESTS ###
test_all:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes


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
