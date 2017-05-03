all: body.md
	showdown makehtml -i body.md -o body.html --github --tables
	@cat header.html > index.html
	@cat body.html >> index.html
	@cat footer.html >> index.html