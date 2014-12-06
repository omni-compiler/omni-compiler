bin_SCRIPTS = bin/C_Back
TARGET = build/om-c-back.jar
SRC    = $(wildcard src/xcodeml/c/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	cd src; $(JAR) cf ../$@ xcodeml/c/*/*.class

$(CLASS): $(SRC)
	$(JAVAC) src/xcodeml/c/*/*.java -g -classpath ../XcodeML-Common/build/om-common.jar -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

.NOTPARALLEL:
