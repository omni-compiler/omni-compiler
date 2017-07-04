/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   varDecl, functionDecl
 *   
 * <br>declarations is converted to XcDeclsObj.
 * <br>but globalDeclarations is converted to XcDeclObj and XcFuncDefObj under XcProgramObj.
 */
public final class XcDeclObj extends XcObj implements XcDecAndDefObj, XcStAndDeclObj
{
    private XcIdent        _ident;

    private XcSourcePosObj _srcPos;

    private XcExprObj      _value;

    private String         _gccAsmCode;

    /**
     * Creates XcDeclObj.
     * 
     * @param ident an identifier declared.
     */
    public XcDeclObj(XcIdent ident)
    {
        _ident = ident;
    }

    /**
     * Creates XcDeclObj.
     * 
     * @param ident an identifier declared.
     * @param value an initial value of the identifier.
     */
    public XcDeclObj(XcIdent ident, XcExprObj value)
    {
        _ident = ident;
        _value = value;
    }

    /**
     * Gets an identifier declared by the object.
     * 
     * @return an identifier declared by the object.
     */
    public XcIdent getIdent()
    {
        return _ident;
    }

    /**
     * Sets an initial value of the identifier.

     * @param expr an initial value of the identifier.
     */
    public void setValue(XcExprObj expr)
    {
        _value = expr;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {
            _value = ((XcExprObj)child);
        } else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_value);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _ident.setValue((XcExprObj)child);
            break;

        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        /*
          append as follows

          #srcpos
          #(type symbol) __asm__(('_gccAsmCode')) = '_value';
         */
        boolean isPreDecl = false;

        w.add(_srcPos);

        _ident.appendDeclCode(w, isPreDecl);

        if(_gccAsmCode != null)
            w.addSpc("__asm__(\"").add(_gccAsmCode).add("\")");

        if(_value != null && !(_value instanceof XcNullExpr))
            w.addSpc("=").addSpc(_value);

        w.eos();
    }

    @Override
    public XcSourcePosObj getSourcePos()
    {
        return _srcPos;
    }

    @Override
    public void setSourcePos(XcSourcePosObj srcPos)
    {
        _srcPos = srcPos;
    }

    /**
     * Sets an assembler code of the identifier. 
     * 
     * @param gccAsmCode an assembler code of the identifier. 
     */
    public void setGccAsmCode(String gccAsmCode)
    {
        _gccAsmCode = gccAsmCode;
    }

    /**
     * Gets a symbol string of the identifier.
     * 
     * @return a symbol string of the identifier.
     */
    public String getSymbol()
    {
        if(_ident == null)
            return null;

        return _ident.getSymbol();
    }
}
