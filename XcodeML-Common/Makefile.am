TARGET = build/om-common.jar
SRC    = $(wildcard src/xcodeml/*.java src/xcodeml/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	mkdir -p build
	cd src;	$(JARCOM) cf ../$@ ./xcodeml/*.class ./xcodeml/*/*.class

SUFFIXES: .class .java

.java.class:
	$(JAVACCOM) $< -classpath ./src -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}
