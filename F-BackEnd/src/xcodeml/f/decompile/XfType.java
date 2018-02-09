/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

/**
 * Type expression in decompiler.
 */
enum XfType
{
    VOID("Fvoid", null, false),
    INT("Fint", "INTEGER", true),
    REAL("Freal", "REAL", true),
    COMPLEX("Fcomplex", "COMPLEX", true),
    LOGICAL("Flogical", "LOGICAL", true),
    CHARACTER("Fcharacter", "CHARACTER", true),
    NUMERIC("Fnumeric", null, true),
    NUMERICALL("FnumericAll", null, true),
    ENUM("FenumType", null, false),
    DERIVED(null, null, false);

    private boolean _isPrimitive = false;
    private String _xcodemlName;
    private String _fortranName;

    private XfType(String xcodemlName, String fortranName, boolean isPrimitive)
    {
        _isPrimitive = isPrimitive;
        _xcodemlName = xcodemlName;
        _fortranName = fortranName;
    }

    public boolean isPrimitive()
    {
        return _isPrimitive;
    }

    public String xcodemlName()
    {
        return _xcodemlName;
    }

    public boolean hasXcodemlName() {
        return _xcodemlName != null;
    }

    public String fortranName()
    {
        return _fortranName;
    }

    public boolean hasFortranName()
    {
        return _fortranName != null;
    }

    public static XfType getTypeIdFromXcodemlTypeName(String xcodemlTypeName)
    {
        if (xcodemlTypeName == null) {
            throw new IllegalArgumentException();
        }

        for (XfType type: XfType.values()) {
            String workTypeName = type.xcodemlName();
            if (workTypeName != null) {
                if (xcodemlTypeName.compareToIgnoreCase(type.xcodemlName()) == 0) {
                    return type;
                }
            }
        }
        return DERIVED;
    }
}

