/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

public class FforAllBlock extends CondBlock
{
    private Xobject ind_var_range;

    public FforAllBlock(BasicBlock cond, Xobject ind_var_range, BlockList body, String construct_name)
    {
        super(Xcode.F_FORALL_STATEMENT, cond, body, construct_name);
        this.ind_var_range = ind_var_range;
    }

    @Override
    public Block copy()
    {
        throw new UnsupportedOperationException(toString());
    }

    @Override
    public BasicBlock getInitBBlock()
    {
        throw new UnsupportedOperationException(toString());
    }

    @Override
    public BasicBlock getIterBBlock()
    {
        throw new UnsupportedOperationException(toString());
    }

    @Override
    public void visitBasicBlock(BasicBlockVisitor v)
    {
        v.visit(bblock);
    }

    public Xobject getInductionVarRange()
    {
        return ind_var_range;
    }

    @Override
    public Xobject toXobject()
    {
        Xobject cond = (bblock != null) ? bblock.toXobject() : null;
        Xobject x = new XobjList(Opcode(), getConstructNameObj());
        for(Xobject a : (XobjList)getInductionVarRange())
        {
          x.add(((XobjList)a).getArg(0));
          x.add(((XobjList)a).getArg(1));
        }
        if (cond != null)
          x.add(cond);
        x.add(getBody().toXobject());
        x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(CondBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(body);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
