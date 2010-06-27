/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import java.util.ArrayList;
import java.util.List;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcIdentTable;
import xcodeml.c.type.XcType;
import xcodeml.c.type.XcTypeEnum;
import xcodeml.c.util.XmcWriter;
import xcodeml.util.XmOption;

/**
 * Internal object represents following elements:
 *   XcodeProgram
 */
public final class XcProgramObj extends XcObj
{
    private List<XcDecAndDefObj> _declAndDefList = new ArrayList<XcDecAndDefObj>();

    private XcIdentTable _identTable;

    private String _language;

    private String _time;

    private String _version;

    private String _source;

    private String _compilerInfo;

    private XcFuncDefObj _xmpMain = null;

    public XcProgramObj()
    {
    }

    public final void setIdentTable(XcIdentTable identTable)
    {
        _identTable = identTable;
    }
    
    public final void addDeclAndDef(XcDecAndDefObj declAndDef)
    {
        _declAndDefList.add(declAndDef);
    }
    
    public final List<XcDecAndDefObj> getDeclAndDefList()
    {
        return _declAndDefList;
    }
    
    public final String getLanguage()
    {
        return _language;
    }
    
    public final void setLanguage(String language)
    {
        _language = language;
    }

    public final String getTime()
    {
        return _time;
    }

    public final void setTime(String time)
    {
        _time = time;
    }

    public final String getVersion()
    {
        return _version;
    }

    public final void setVersion(String version)
    {
        _version = version;
    }

    public final String getSource()
    {
        return _source;
    }

    public final void setSource(String source)
    {
        _source = source;
    }

    public final String getCompilerInfo()
    {
        return _compilerInfo;
    }

    public final void setCompilerInfo(String compilerInfo)
    {
        _compilerInfo = compilerInfo;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcDecAndDefObj)
            addDeclAndDef((XcDecAndDefObj)child);
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
        if(_declAndDefList.isEmpty())
            return null;
        return _declAndDefList.toArray(new XcNode[_declAndDefList.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if(index < _declAndDefList.size())
            _declAndDefList.set(index, (XcDecAndDefObj)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    private void _renameXmpMain()
    {
        XcFuncDefObj fdef = _findMainFunc();

        if(fdef == null)
            return;

        List<XcIdent> coArrayIdentList = new ArrayList<XcIdent>();

        for(XcDecAndDefObj decf : _declAndDefList) {
            if((decf instanceof XcDeclObj) == false)
                continue;

            XcDeclObj decl = (XcDeclObj)decf;

            XcIdent ident = decl.getIdent();
            XcType type = ident.getType();

            if(type.getTypeEnum() == XcTypeEnum.COARRAY)
                coArrayIdentList.add(ident);
        }

        XcXmpFactory.renameMain(fdef);
    }

    private XcFuncDefObj _findMainFunc()
    {
        for(XcDecAndDefObj decf : _declAndDefList) {
            if((decf instanceof XcFuncDefObj) == false)
                continue;

            XcFuncDefObj fdef = (XcFuncDefObj)decf;

            if(fdef.isMain()) {
                return fdef;
            }
        }

        return null;
    }

    public final void writeTo(XmcWriter w) throws XmException
    {
        w.add("/*").lf();
        w.add(" * Original Source  : ").add(_source).lf();
        w.add(" * Language         : ").add(_language).lf();
        w.add(" * Compiled Time    : ").add(_time).lf();
        w.add(" * Compiler Info    : ").add(_compilerInfo).lf();
        w.add(" * Compiler Version : ").add(_version).lf();
        w.add(" */").lf();

        appendCode(w);
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        //if(XmOption.isXcalableMP())
        //    _renameXmpMain();

        //if(XmOption.isXcalableMP())
        //    w.add(XcXmpFactory.createXmpIncludeDirective());

        if(XmOption.isSuppressLineDirective() == false) {
            w.noLfOrLf().add("# 1 ").add("\"").add(_source).add("\"").lf();
        }

        if(_identTable != null) {
            _identTable.appendCode(w, _declAndDefList);
        } else {
            for(XcDecAndDefObj declAndDef : _declAndDefList)
                w.add(declAndDef);
        }

        //if(XmOption.isXcalableMP())
        //    w.add(_xmpMain);
    }

    final void reduce()
    {
        _reduce(this);
    }

    private static XcNode _reduce(XcNode node)
    {
        if(node == null)
            return null;

        XcNode[] children = node.getChild();

        if(children != null) {
            for(int i = 0; i < children.length; ++i) {
                XcNode child = children[i];
                XcNode newChild = _reduce(child);
                if(newChild != null)
                    node.setChild(i, newChild);
            }
        }

        if(node instanceof XcExprObj)
            return _reduceExprObj((XcExprObj)node);

        return null;
    }

    private static final XcNode _reduceExprObj(XcExprObj expr)
    {
        if(expr instanceof XcRefObj.PointerRef) {
            XcRefObj.PointerRef ptrRef = (XcRefObj.PointerRef)expr;
            if(ptrRef.getExpr() instanceof XcRefObj.Addr) {
                XcRefObj.Addr addr = (XcRefObj.Addr)ptrRef.getExpr();
                return addr.getExpr();
            }
        }

        return null;
    }
}
