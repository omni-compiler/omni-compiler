TARGET = build/om-exc-tools.jar
SRC    = $(wildcard src/exc/*/*.java)

all-local: $(TARGET)

$(TARGET): $(SRC)
	$(JAVAC) -encoding utf8 src/exc/*/*.java -classpath ../XcodeML-Common/build/om-common.jar:../C-BackEnd/build/om-c-back.jar:./src -Xlint:deprecation
	mkdir -p build
	cd src; $(JAR) cf ../$@ ./exc/*/*.class

clean-local:
	rm -f $(TARGET)
	rm -f src/exc/*/*.class

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

