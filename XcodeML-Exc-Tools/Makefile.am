TARGET = build/om-exc-tools.jar
SRC    = $(wildcard src/exc/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	cd src; $(JARCOM) cf ../$@ ./exc/*/*.class

$(CLASS): $(SRC)
	$(JAVACCOM) -g src/exc/*/*.java -classpath ../XcodeML-Common/build/om-common.jar:../C-BackEnd/build/om-c-back.jar -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

.NOTPARALLEL:
