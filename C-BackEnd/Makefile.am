bin_SCRIPTS = bin/C_Back
TARGET = build/om-c-back.jar
SRC    = $(wildcard src/xcodeml/c/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	mkdir -p build
	cd src; $(JAR) cf ../$@ xcodeml/c/*/*.class

SUFFIXES: .class .java

.java.class:
	$(JAVAC) $< -classpath ../XcodeML-Common/build/om-common.jar:./src -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

