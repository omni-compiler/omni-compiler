package xcodeml.c.type;

import xcodeml.util.XmException;
import xcodeml.c.util.XmcWriter;

/**
 * type of array
 */
public final class XcArrayType extends XcArrayLikeType
{
    /* type qualifier: static */
    private boolean _isStatic;

    public final boolean isStatic()
    {
        return _isStatic;
    }

    public final void setIsStatic(boolean enable)
    {
        _isStatic = enable;
    }

    public XcArrayType(String typeId)
    {
        super(XcTypeEnum.ARRAY, typeId);
    }

    public final void appendArraySpecCode(
        XmcWriter w, boolean isPreDecl, boolean isFirstIndex)
        throws XmException
    {
        w.add("[");

        this.appendTypeQualCode(w);

        if(_isStatic)
            w.addSpc("static");

        if(isArraySizeExpr() && (getArraySizeExpr() != null)) {
            if(isPreDecl) {
                if(isFirstIndex)
                    w.addSpc("*");
            } else {
                w.addSpc(getArraySizeExpr());
            }
        } else if(isArraySize()){
            w.addSpc(getArraySize());
        }

        w.add("]");
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(128);
        b.append("[");
        commonToString(b);
        b.append("arraySize=").append(getArraySize()).append("]");
        return b.toString();
    }
}
