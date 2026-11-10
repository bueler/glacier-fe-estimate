all: estimate.pdf

_figures := stokesdomain giscross extruded filled icefree
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/stokesdomain.pdf: genfigs/stokesdomain.py
	(cd genfigs/ && python3 stokesdomain.py)

genfigs/giscross.pdf: genfigs/giscross.py
	(cd genfigs/ && python3 giscross.py)

genfigs/extruded.pdf: genfigs/extruded.py
	(cd genfigs/ && python3 extruded.py)

genfigs/filled.pdf genfigs/icefree.pdf: genfigs/noncoercive.py
	(cd genfigs/ && python3 noncoercive.py)

estimate.pdf: estimate.tex estimate.aux estimate.bbl
	pdflatex estimate
	pdflatex estimate

estimate.aux: estimate.tex cropfigures
	pdflatex estimate
	bibtex estimate

estimate.bbl: estimate.aux estimate.bib
	bibtex estimate

# latexdiff against submitted version; --math-markup=off avoids ulem/\left..\right
# breakage, and \wSIA was defined in the body of assubmitted.tex
diff.pdf: assubmitted.tex estimate.tex estimate.bib cropfigures
	latexdiff --math-markup=off --config="MATHARRREPL=align*" assubmitted.tex estimate.tex > diff.tex
	sed -i 's/^\\begin{document}/\\providecommand{\\wSIA}{\\tilde w}\n\\begin{document}/' diff.tex
	pdflatex diff
	bibtex diff
	pdflatex diff
	pdflatex diff

# generate .zip of source files suitable for submitting to arxiv
publish.zip: estimate.pdf estimate.bbl
	rm -rf publish/
	mkdir publish/
	mkdir publish/figs/
	cp -f figs/*.png publish/figs/
	mkdir publish/genfigs/
	cp -f genfigs/*.pdf publish/genfigs/
	mkdir publish/genfigs/gis/
	cp -f genfigs/gis/*.png publish/genfigs/gis/
	mkdir publish/tikz/
	cp -f tikz/*.tex publish/tikz/
	cp -f siamart250211.cls siamplain.bst estimate.bib estimate.tex publish/
	zip -r publish publish

.PHONY: clean cropfigures _figures

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
	@rm -rf genfigs/*.pdf genfigs/__pycache__/
	@rm -rf publish*
