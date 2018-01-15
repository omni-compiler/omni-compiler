package exc.block;

import exc.object.*;

public class FforAllBlock extends CondBlock
{
    private Xtype type;
    private Xobject ind_var_range;

    public FforAllBlock(Xtype type, BasicBlock cond, Xobject ind_var_range, BlockList body, String construct_name)
    {
        super(Xcode.F_FORALL_STATEMENT, cond, body, construct_name);
        this.type = type;
        this.ind_var_range = ind_var_range;
    }

    public Xtype getType()
    {
        return type;
    }

    public void setType(Xtype type)
    {
        this.type = type;
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
        Xobject x = new XobjList(Opcode(), getType(), getConstructNameObj());
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
        s.append("(FforAllBlock construct_name[");
        s.append((getConstructNameObj() != null) ? getConstructNameObj() : "(no name)");
        s.append("] type[");
        s.append(getType());
        s.append("] index_range[");
        s.append(getInductionVarRange());
        s.append("] condition[");
        s.append((getBasicBlock() != null) ? getBasicBlock().getExpr() : "(no condition)");
        s.append("] body[");
        s.append(body);
        s.append("])");
        return s.toString();
    }
}
