/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

/**
 * Symbol expression in decompiler.
 */
class XfSymbol
{
    private String _symbolName;
    private XfType _typeId;
    private String _derivedName;

    public XfSymbol(String symbolName)
    {
        this(symbolName, XfType.VOID, null);
    }

    public XfSymbol(String symbolName, XfType typeId)
    {
        this(symbolName, typeId, null);
    }

    public XfSymbol(String symbolName, XfType typeId, String derivedName)
    {
        _symbolName = symbolName;
        _typeId = typeId;
        _derivedName = derivedName;
    }

    public XfType getTypeId()
    {
        return _typeId;
    }

    public String getSymbolName()
    {
        return _symbolName;
    }

    public String getDerivedName()
    {
        return _derivedName;
    }
}

