package xcodeml.c.util;

import xcodeml.c.type.XcIntegerType;
import xcodeml.c.type.XcPointerType;

/**
 * Utilities of XmOject represents type.
 */
public class XmcTypeUtil
{
    private XmcTypeUtil()
    {
    }

    /**
     * Creates XmObject represent type 'const int * const'
     *
     * @return internal object represent 'const int * const' type.
     */
    public static XcPointerType createConstCharConstPointer()
    {
        XcPointerType ptr = new XcPointerType();
        ptr.setIsConst(true);
        XcIntegerType.Char chr = new XcIntegerType.Char();
        chr.setIsConst(true);
        ptr.setRefType(chr);
        return ptr;
    }
}
