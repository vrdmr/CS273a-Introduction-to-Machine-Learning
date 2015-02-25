.PHONY: dist clean
GIT ?= git
VER = 1.1

dist:
	git archive --format=tar --prefix=svmdemo/ v${VER} | gzip > svmdemo-$(VER).tar.gz

clean:
	echo rm -f `find . -name '*~'`
	rm -f svmdemo-*.tar.gz