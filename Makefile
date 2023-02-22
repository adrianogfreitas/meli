.PHONY: venv jupyter-kernel install init

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = meli
PYTHON_INTERPRETER = python3
VENV_NAME = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create a virtual environment
venv:
	$(PYTHON_INTERPRETER) -m pip install virtualenv
	virtualenv -p $(PYTHON_INTERPRETER) $(VENV_NAME)
	
## Install requirements - for local dev - includes dev requirements
install:
	. ./$(VENV_NAME)/bin/activate && \
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]" && \
	$(PYTHON_INTERPRETER) -m nltk.downloader stopwords && \
	$(PYTHON_INTERPRETER) -m nltk.downloader wordnet

## Add virtualenv to ipython kernel - for jupyter notebooks
jupyter-kernel:
	. ./$(VENV_NAME)/bin/activate && \
	$(PYTHON_INTERPRETER) -m ipykernel install --name=$(PROJECT_NAME) --user

## Run on the first execution of this project
init: venv install jupyter-kernel

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
