bin_SCRIPTS = bin/F_Back bin/F_Trans bin/F_Decomp
TARGET      = build/om-f-back.jar
SRC         = $(wildcard src/xcodeml/f/*/*.java)
CLASS       = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	cd src; $(JAR) cf ../$@ xcodeml/f/*/*.class

$(CLASS): $(SRC)
	$(JAVAC) src/xcodeml/f/*/*.java -g -classpath ../XcodeML-Common/build/om-common.jar -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

.NOTPARALLEL: