package xcodeml.c.decompile;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdentTable;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   compoundStatment
 */
public class XcCompStmtObj extends XcStmtObj
{
    private List<XcStAndDeclObj> _stmtAndDeclList = new ArrayList<XcStAndDeclObj>();

    private XcDeclsObj _decls;

    private XcIdentTable _identTable;

    /**
     *  Creates a XcCompStmtObj.
     */
    public XcCompStmtObj()
    {
    }

    /**
     *  Sets a identifier table to the object.
     *  
     *  @param identTable an identifier table of compound statement.
     */
    public final void setIdentTable(XcIdentTable identTable)
    {
        _identTable = identTable;
    }

    /**
     *  Geta identifier table to the object.
     *  
     *  @return identTable an identifier table of compound statement.
     */
    public final XcIdentTable getIdentTable()
    {
        return _identTable;
    }

    /**
     * Adds declaration to the compound statement object.
     * 
     * @param decl a declaration.
     */
    public final void addDecl(XcDeclObj decl)
    {
        if(_decls == null)
            _decls = new XcDeclsObj();

        _decls.add(decl);
    }

    /**
     * Adds declaration or statement to the compound statement object.
     * 
     * @param stmt a declaration or a statement.
     */
    public final void addStmtAndDecl(XcStAndDeclObj stmt)
    {
        _stmtAndDeclList.add(stmt);
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcDeclsObj)
            _decls = (XcDeclsObj)child;
        else if(child instanceof XcStAndDeclObj)
            _stmtAndDeclList.add((XcStAndDeclObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        if(_stmtAndDeclList.isEmpty())
            return null;
        return _stmtAndDeclList.toArray(new XcNode[_stmtAndDeclList.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if(index < _stmtAndDeclList.size())
            _stmtAndDeclList.set(index, (XcStAndDeclObj)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    /**
     * Appends contents of compound statement with XmWriter.
     * 
     * @param w a writer appends the code.
     * @param insideCompExpr true if is the object is the term of the compound expression. 
     * @throws XmException thrown if a writer failed to append the content.
     */
    public final void appendContent(XmcWriter w, boolean insideCompExpr) throws XmException
    {
        if(_identTable == null) {
            w.add(_decls);
        } else {
            _identTable.appendCode(w, _decls);
        }

        Iterator<XcStAndDeclObj> iter = _stmtAndDeclList.iterator();

        XcStAndDeclObj stmtAndDecl = null;

        while(iter.hasNext() == true) {
            stmtAndDecl = iter.next();
            w.add(stmtAndDecl);

            if(insideCompExpr == false)
                w.noLfOrLf();
        }

        if(stmtAndDecl != null &&
           ((stmtAndDecl instanceof XcControlStmtObj.CaseLabel) ||
            (stmtAndDecl instanceof XcControlStmtObj.GccRangedCaseLabel) ||
            (stmtAndDecl instanceof XcControlStmtObj.Label) ||
            (stmtAndDecl instanceof XcControlStmtObj.DefaultLabel))
           )
            w.eos();
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.add("{").lf();
        appendContent(w, false);
        w.add("}").lf();
    }
}
