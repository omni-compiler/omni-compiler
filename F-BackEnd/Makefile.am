TARGET      = build/om-f-back.jar
SRC         = $(wildcard src/xcodeml/f/*/*.java)

all-local: $(TARGET)

$(TARGET): $(SRC)
	$(JAVAC) -encoding utf8 src/xcodeml/f/*/*.java -classpath ../XcodeML-Common/build/om-common.jar:./src -Xlint:deprecation
	$(MKDIR_P) build
	cd src; $(JAR) cf ../$@ xcodeml/f/*/*.class

clean-local:
	rm -f $(TARGET)
	rm -f src/xcodeml/f/*/*.class

install: $(TARGET)
	$(MKDIR_P) $(DESTDIR)${javadir}
	$(INSTALL) $^ $(DESTDIR)${javadir}
