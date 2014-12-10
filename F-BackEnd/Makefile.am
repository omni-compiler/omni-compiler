bin_SCRIPTS = bin/F_Back bin/F_Trans bin/F_Decomp
TARGET      = build/om-f-back.jar
SRC         = $(wildcard src/xcodeml/f/*/*.java)
CLASS       = $(SRC:.java=.class)

all-local: $(TARGET)

$(TARGET): $(CLASS)
	mkdir -p build
	cd src; $(JARCOM) cf ../$@ xcodeml/f/*/*.class

.java.class:
	$(JAVACCOM) $< -classpath ../XcodeML-Common/build/om-common.jar:./src -Xlint:deprecation

clean-local:
	rm -f $(TARGET) $(CLASS)

install: $(TARGET)
	mkdir -p ${javadir}
	$(INSTALL) $^ ${javadir}
