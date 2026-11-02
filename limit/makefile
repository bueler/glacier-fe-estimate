all: limit.pdf

limit.pdf: limit.tex limit.aux limit.bbl
	pdflatex limit
	pdflatex limit

limit.aux: limit.tex
	pdflatex limit
	bibtex limit

limit.bbl: limit.aux limit.bib
	bibtex limit

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
