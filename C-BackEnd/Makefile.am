TARGET = build/om-c-back.jar
SRC    = $(wildcard src/xcodeml/c/*/*.java)

all-local: $(TARGET)

$(TARGET): $(SRC)
	$(JAVAC) src/xcodeml/c/*/*.java -classpath ../XcodeML-Common/build/om-common.jar:./src -Xlint:deprecation
	mkdir -p build
	cd src; $(JAR) cf ../$@ xcodeml/c/*/*.class

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

