/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   gccAsmDefinition
 */
public final class XcGccAsmDefinition extends XcObj implements XcDecAndDefObj
{
    private XcConstObj.StringConst _asmCode;

    private boolean _isGccExtension;

    /**
     * Creates XcGccAsmDefinition.
     */
    public XcGccAsmDefinition()
    {
    }

    /**
     * Sets if is the asm code used with __extension__.
     *  
     * @param isGccExtension true if the asm code is used with __extension__.
     */
    public void setIsGccExtension(boolean isGccExtension)
    {
        _isGccExtension = isGccExtension;
    }

    /**
     * Tests if is the asm code used with __extension__.
     * 
     * @return true if the asm code is used with __extension__.
     */
    public boolean isGccExtension()
    {
        return _isGccExtension;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcConstObj.StringConst)
            _asmCode = (XcConstObj.StringConst)child;
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return null;
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        if(_isGccExtension)
            w.addSpc("__extension__");

        w.addSpc("__asm__(").add(_asmCode).add(")").eos();
    }
}
