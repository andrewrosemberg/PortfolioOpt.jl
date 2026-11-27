# Builds the formulation diagram from formulation.tex: PDF with pdflatex, SVG with
# pdftocairo, then a white rectangle under the drawing so it reads on dark pages too.
#
#   make -C docs/src/assets/diagrams/src
#
# Needs a TeX Live with pdflatex and poppler's pdftocairo.
OUT := ..
TMP := build
NAME := portfolioopt--formulation

all: $(OUT)/$(NAME).svg $(OUT)/$(NAME)-compact.svg

$(TMP)/%.pdf: formulation.tex
	mkdir -p $(TMP)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(TMP) -jobname=$* formulation.tex >/dev/null

$(OUT)/%.svg: $(TMP)/%.pdf
	pdftocairo -svg $< $(TMP)/$*.raw.svg
	perl -0pe 's/(<svg [^>]*>)/$$1\n<rect width="100%" height="100%" fill="#ffffff"\/>/' $(TMP)/$*.raw.svg > $@

clean:
	rm -rf $(TMP)

.PHONY: all clean
