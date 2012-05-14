/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.Xobject;

/**
 * Represents Fortran do statement block.
 */
public class FdoBlock extends Block implements ForBlock
{
    private BasicBlock ind_var_part, lower_part, upper_part, step_part;
    private BlockList body;
    private boolean is_canonical;
    
    public FdoBlock(Xobject ind_var, Xobject idx_range, BlockList body, String construct_name)
    {
        super(Xcode.F_DO_STATEMENT, null, construct_name);
        this.body = body;
        this.is_canonical = false;
        body.parent = this;
        ind_var_part = BasicBlock.Expr(this, ind_var);
        
        if(idx_range != null) {
            lower_part = BasicBlock.Expr(this, idx_range.getArg(0));
            upper_part = BasicBlock.Expr(this, idx_range.getArg(1));
            step_part = BasicBlock.Expr(this, idx_range.getArgOrNull(2));
        }
    }

    @Override
    public void Canonicalize()
    {
        is_canonical = true;
    }

    @Override
    public Xcode getCheckOpcode()
    {
        return Xcode.LOG_NOT_EXPR;
    }

    @Override
    public Xobject getInductionVar()
    {
        return ind_var_part.getExpr();
    }

    @Override
    public Xobject getLowerBound()
    {
        return lower_part.getExpr();
    }

    @Override
    public Xobject getStep()
    {
        return step_part.getExpr();
    }

    @Override
    public Xobject getUpperBound()
    {
        return upper_part.getExpr();
    }

    @Override
    public boolean isCanonical()
    {
        return is_canonical;
    }

    @Override
    public void setLowerBound(Xobject x)
    {
        lower_part.setExpr(x);
    }

    @Override
    public void setStep(Xobject x)
    {
        step_part.setExpr(x);
    }

    @Override
    public void setUpperBound(Xobject x)
    {
        upper_part.setExpr(x);
    }

    @Override
    public BlockList getBody()
    {
        return body;
    }

    @Override
    public void setBody(BlockList s)
    {
        body = s;
        s.parent = this;
    }

    @Override
    public BasicBlock getInitBBlock()
    {
        return null;
    }

    @Override
    public BasicBlock getIterBBlock()
    {
        return null;
    }
    
    @Override
    public void visitBasicBlock(BasicBlockVisitor v)
    {
        v.visit(ind_var_part);
        v.visit(lower_part);
        v.visit(upper_part);
        v.visit(step_part);
        v.visit(bblock);
    }

  // only for FdoBlock
    public void setInductionVar(Xobject ind_var)
    {
        ind_var_part.setExpr(ind_var);
    }

    @Override
    public Xobject toXobject()
    {
        return Xcons.List(Opcode(),
            getConstructNameObj(),
            (ind_var_part != null ? ind_var_part.getExpr() : null),
            (lower_part != null ? Xcons.List(Xcode.F_INDEX_RANGE,
                getLowerBound(), getUpperBound(), getStep()) : null),
            body.toXobject());
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(FdoBlock ");
        s.append("(");
        s.append(getInductionVar());
        s.append(" ");
        s.append(getLowerBound());
        s.append(" ");
        s.append(getUpperBound());
        s.append(" ");
        s.append(getStep());
        s.append(")");
        s.append(" ");
        s.append(body);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
