TARGET = build/om-common.jar
SRC    = $(wildcard src/xcodeml/*.java src/xcodeml/*/*.java)

all-local: $(TARGET)

$(TARGET): $(SRC)
	$(JAVAC) -encoding utf8 src/xcodeml/*.java src/xcodeml/*/*.java -classpath ./src -Xlint:deprecation
	mkdir -p build
	cd src;	$(JAR) cf ../$@ ./xcodeml/*.class ./xcodeml/*/*.class

clean-local:
	rm -f $(TARGET)
	rm -f src/xcodeml/*.class
	rm -f src/xcodeml/*/*.class

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}
