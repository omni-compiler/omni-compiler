TARGET      = build/om-f-back.jar
SRC         = $(wildcard src/xcodeml/f/*/*.java)

all-local: $(TARGET)

$(TARGET): $(SRC)
	$(JAVAC) src/xcodeml/f/*/*.java -classpath ../XcodeML-Common/build/om-common.jar:./src -Xlint:deprecation
	mkdir -p build
	cd src; $(JAR) cf ../$@ xcodeml/f/*/*.class

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}
