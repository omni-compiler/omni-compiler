TARGET = build/om-exc-tools.jar
SRC    = $(wildcard src/exc/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	mkdir -p build
	cd src; $(JAR) cf ../$@ ./exc/*/*.class

SUFFIXES: .class .java

.java.class:
	$(JAVAC) $< -classpath ../XcodeML-Common/build/om-common.jar:../C-BackEnd/build/om-c-back.jar:./src -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

