all: unique.pdf

unique.pdf: unique.tex unique.aux unique.bbl
	pdflatex unique
	pdflatex unique

unique.aux: unique.tex
	pdflatex unique
	bibtex unique

unique.bbl: unique.aux unique.bib
	bibtex unique

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
