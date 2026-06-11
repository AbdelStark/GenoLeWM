# GenoLeWM paper -- build with tectonic (self-contained, fetches packages).
PAPER = main

.PHONY: all clean veryclean

all: $(PAPER).pdf

# Two passes so forward references to floats (figures/tables) resolve,
# with intermediates kept between passes.
$(PAPER).pdf: $(PAPER).tex figures.tex tables.tex refs.bib neurips.sty
	tectonic --keep-intermediates $(PAPER).tex
	tectonic --keep-intermediates $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).out $(PAPER).log

veryclean: clean
	rm -f $(PAPER).pdf
