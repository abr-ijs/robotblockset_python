# Minimal makefile for Sphinx documentation

SPHINXBUILD   ?= sphinx-build
SPHINXAUTOGEN ?= sphinx-autogen
SOURCEDIR     = .
BUILDDIR      = _build
APIDOCINDEX   = api/index.rst
APIGENDIR     = api/generated

.PHONY: html clean

html:
	rm -rf "$(APIGENDIR)"
	mkdir -p "$(APIGENDIR)"
	$(SPHINXAUTOGEN) -t "$(SOURCEDIR)/_templates" -o "$(APIGENDIR)" "$(APIDOCINDEX)"
	$(SPHINXAUTOGEN) -t "$(SOURCEDIR)/_templates" -o "$(APIGENDIR)" "$(APIGENDIR)"/*.rst
	$(SPHINXAUTOGEN) -t "$(SOURCEDIR)/_templates" -o "$(APIGENDIR)" "$(APIGENDIR)"/*.rst
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(APIGENDIR)"
	rm -rf "$(BUILDDIR)"
