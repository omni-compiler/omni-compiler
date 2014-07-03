# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
java_DATA = build/om-common.jar
ANT = @ANTCOM@ -buildfile build.xml
MKDIR_P = @MKDIR_P@
SRC = $(wildcard src/*/*.java) $(wildcard src/*/*/*.java)

all-local: $(java_DATA)

$(java_DATA): $(SRC)
	$(ANT) jar

jar:
	$(ANT) jar

jar-test:
	$(ANT) jar-test

clean-local:
	$(ANT) clean

html-local:
	$(ANT) document
