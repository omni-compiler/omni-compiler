package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcFuncType;
import xcodeml.c.type.XcGccAttributable;
import xcodeml.c.type.XcGccAttributeList;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcParamList;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   functionDefinition
 */
public class XcFuncDefObj extends XcObj
    implements XcDecAndDefObj, XcGccAttributable, XcSourcePositioned
{
    private XcIdent        _ident;
    private XcCompStmtObj  _compStmt;
    private boolean        _isGccExtension;
    private XcSourcePosObj _srcPos;

    /* types of parameters */
    private XcParamList _paramList = new XcParamList();

    /* GCC attribute */
    private XcGccAttributeList _gccAttrs;

    /**
     * Tests if parameters used with an ellipsis.
      *
     * @return true if parameters used with an ellipsis.
     */
    public boolean isEllipsised()
    {
        return _paramList.isEllipsised();
    }

    /**
     * Sets parameters to be used with an ellipsis.
      *
     * @return true if parameters used with an ellipsis.
     */
    public void setIsEllipsised(boolean enable)
    {
        _paramList.setIsEllipsised(enable);
    }

    /**
     * Adds an identifier as a parameter.
     * 
     * @param param an identifier is to be a parameter.
     */
    public final void addParam(XcIdent param)
    {
        _paramList.add(param);
    }
    
    /**
     * Creates XcFuncDefObj.
     * 
     * @param ident an identifier of the function.
     */
    public XcFuncDefObj(XcIdent ident)
    {
        _ident = ident;
    }
    
    /**
     * Creates XcFuncDefObj.
     * 
     * @param ident an identifier of the function.
     * @param stmt a body of the function.
     */
    public XcFuncDefObj(XcIdent ident, XcCompStmtObj stmt)
    {
        _ident = ident;
        _compStmt = stmt;
    }

    /**
     * Gets a source position object.
     * 
     * @return a source position object.
     */
    public final XcSourcePosObj getSourcePos()
    {
        return _srcPos;
    }

    /**
     * Sets a source position object.
     * 
     * @param srcPos source position object.
     */
    public final void setSourcePos(XcSourcePosObj srcPos)
    {
        _srcPos = srcPos;
    }

    /**
     * Gets an identifier of the function which the object defined.
     */
    public final XcIdent getIdent()
    {
        return _ident;
    }

    /**
     * Gets a body of the function which the object defined.
     */
    public final XcCompStmtObj getCompStmt()
    {
        return _compStmt;
    }

    /**
     * Gets a parameter list of the function which the object defined.
     */
    public final XcParamList getParamList()
    {
        return _paramList;
    }

    /**
     * Sets if is the function defined with __extension__.
     *  
     * @param isGccExtension true if the function is defined with __extension__
     */
    public final void setIsGccExtension(boolean isGccExtension)
    {
        _isGccExtension = isGccExtension;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcCompStmtObj) {
            if (_compStmt == null) {
                _compStmt = (XcCompStmtObj)child;
            } else {
                _compStmt.addChild(child);
            }
        } else if(child instanceof XcStmtObj
                || child instanceof XcFuncDefObj) {
            if(_compStmt == null) {
                _compStmt = new XcCompStmtObj();
            }
            _compStmt.addChild(child);
        } else if(child instanceof XcGccAttributeList) {
            _gccAttrs = (XcGccAttributeList)child;
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
        return toNodeArray(_compStmt);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _compStmt = (XcCompStmtObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        /* functionDefition is not predeclaration of function */
        boolean isPreDecl = false;
        XcParamList paramList;
        XcFuncType ft = (XcFuncType)_ident.getType();

        w.add(_srcPos);
        isPreDecl = ft.isPreDecl();
        ft.setIsPreDecl(_compStmt == null);
        paramList = ft.getParamList();
        ft.setParamList(_paramList);

        if (_isGccExtension)
            w.add("__extension__ ");

        _ident.appendFuncDeclCode(w, false, _gccAttrs);

        ft.setParamList(paramList);
        ft.setIsPreDecl(isPreDecl);

        if(_compStmt != null) {
            w.lf().add(_compStmt);
        } else {
            w.lf().add("{").lf().add("}").lf();
        }
    }

    @Override
    public void setGccAttribute(XcGccAttributeList attrs)
    {
        _gccAttrs = attrs;
    }

    @Override
    public XcGccAttributeList getGccAttribute()
    {
        return _gccAttrs;
    }

    /**
     * Gets a symbol string of indentifier.
     * 
     * @return a symbol string of indentifier.
     */
    public String getSymbol()
    {
        if(_ident == null)
            return null;

        return _ident.getSymbol();
    }

    /**
     * Sets a symbol string of indentifier.
     * 
     * @param symbol a symbol string of indentifier.
     */
    public void setSymbol(String symbol)
    {
        if(_ident == null)
            return;

        _ident.setSymbol(symbol);
    }

    /**
     * Tests if is the function a main function.
     * 
     * @return true if the function is a main function.
     */
    public boolean isMain()
    {
        if(_ident == null)
            return false;

        String symbol = _ident.getSymbol();
        return symbol.equals("main");
    }

}
