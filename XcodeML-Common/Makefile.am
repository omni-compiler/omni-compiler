TARGET = build/om-common.jar
SRC    = $(wildcard src/xcodeml/*.java src/xcodeml/*/*.java)
CLASS  = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	cd src;	$(JARCOM) cf ../$@ ./xcodeml/*.class ./xcodeml/*/*.class

$(CLASS): $(SRC)
	$(JAVACCOM) -g src/xcodeml/*.java src/xcodeml/*/*.java -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}

.NOTPARALLEL: