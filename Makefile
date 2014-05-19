article.pdf: article.tex paper.bib
	pdflatex article
	bibtex article
	pdflatex article
	pdflatex article
