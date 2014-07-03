# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
bin_SCRIPTS=bin/F_Back bin/F_Trans bin/F_Decomp
java_DATA = build/om-f-back.jar
ANT = @ANTCOM@ -buildfile build.xml
MKDIR_P = @MKDIR_P@
SRC = $(wildcard *.java)

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
