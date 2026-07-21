NAME := layer

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex $(NAME).aux $(NAME).bbl
	pdflatex $(NAME)
	pdflatex $(NAME)

$(NAME).aux: $(NAME).tex
	pdflatex $(NAME)
	bibtex $(NAME)

$(NAME).bbl: $(NAME).aux $(NAME).bib
	bibtex $(NAME)

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
