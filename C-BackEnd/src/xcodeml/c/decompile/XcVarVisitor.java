/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import java.util.Set;

import xcodeml.XmObj;
import xcodeml.c.binding.XcScanningVisitor;
import xcodeml.c.binding.gen.*;

/**
 * Visitor search a XmObj tree for variables reference.
 */
public class XcVarVisitor extends XcScanningVisitor
{
    private Set<String> _variables = null;

    /**
     * Checks if a XmObj tree contains variable references.
     * 
     * @param visitable a XmObj tree.
     * @return true if the tree contains variable references.
     */
    public static boolean checkContainVar(XmObj visitable)
    {
        XcVarVisitor visitor = new XcVarVisitor();

        return ((IRVisitable)visitable).enter(visitor);
    }

    /**
     * Checks if a XmObj tree contains variable references.
     * 
     * @param visitable a XmObj tree.
     * @return true if the tree contains variable references.
     */
    public static boolean checkContainVar(IRVisitable visitable)
    {
        XcVarVisitor visitor = new XcVarVisitor();

        return visitable.enter(visitor);
    }

    /**
     * Creates a XcVarVisitor.
     */
    public XcVarVisitor()
    {
    }

    /**
     * Creates a XcVarVisitor.
     * 
     * @param variables a set of variable's name a visitor use.
     */
    public XcVarVisitor(Set<String> variables)
    {
        _variables = variables;
    }

    @Override
    public boolean enter(XbcVar visitable)
    {
        if(_variables != null)
            _variables.add(visitable.getContent());

        return true;
    }

    @Override
    public boolean enter(XbcVarAddr visitable)
    {
        if(_variables != null)
            _variables.add(visitable.getContent());

        return true;
    }

    @Override
    public boolean enter(XbcFuncAddr visitable)
    {
        if(_variables != null)
            _variables.add(visitable.getContent());

        return true;
    }

//     @Override
//     public boolean enter(XbcArrayRef visitable)
//     {
//         if(_variables != null)
//             _variables.add(visitable.getContent());

//         return true;
//     }

    @Override
    public boolean enter(XbcMemberRef visitable)
    {
        if(_variables != null)
            _enterChildren(visitable);

        return true;
    }

    @Override
    public boolean enter(XbcMemberAddr visitable)
    {
        _enterChildren(visitable);

        return true;
    }

    @Override
    public boolean enter(XbcMemberArrayRef visitable)
    {
        _enterChildren(visitable);

        return true;
    }

    @Override
    public boolean enter(XbcMemberArrayAddr visitable)
    {
        _enterChildren(visitable);

        return true;
    }
}
