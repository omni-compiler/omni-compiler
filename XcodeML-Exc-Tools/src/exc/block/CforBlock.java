/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

public class CforBlock extends CondBlock implements ForBlock
{
    private BasicBlock init_part, iter_part;

    // for cannonical shaped for-loop
    private boolean is_canonical;
    private Xobject ind_var;
    private Xobject lowerb, upperb, step;
    private Xobject min_upperb;

    public CforBlock(BasicBlock init, BasicBlock cond, BasicBlock iter, BlockList body, String construct_name)
    {
        super(Xcode.FOR_STATEMENT, cond, body, construct_name);
        this.is_canonical = false;
        this.init_part = init;
        this.iter_part = iter;
        if(init_part != null)
            init_part.parent = this;
        if(iter_part != null)
            iter_part.parent = this;
    }
    
    public CforBlock(CforBlock forBlock)
    {
        super(forBlock);
          this.is_canonical = forBlock.is_canonical;
            if(forBlock.init_part != null){
              this.init_part = forBlock.init_part.copy();
              this.init_part.parent = this;
            }
            if(forBlock.iter_part != null){
              this.iter_part = forBlock.iter_part.copy();
              this.iter_part.parent = this;
            }
    }
      
    @Override
    public Block copy()
    {
        return new CforBlock(this);
    }

    @Override
    public BasicBlock getInitBBlock()
    {
        return init_part;
    }

    @Override
    public BasicBlock getIterBBlock()
    {
        return iter_part;
    }

    public void setInitBlock(BasicBlock s)
    {
        init_part = s;
        s.parent = this;
    }

    public void setIterBlock(BasicBlock s)
    {
        iter_part = s;
        s.parent = this;
    }

    @Override
    public void visitBasicBlock(BasicBlockVisitor v)
    {
        v.visit(init_part);
        v.visit(bblock);
        v.visit(iter_part);
    }

    // used by Canonicalize()
    private Xobject getTranslatedForBlockIndVarDecl(String indVarName)
    {
        Xobject decl = null;
        BlockList bl = this.getParent();
        if ((decl = bl.findLocalDecl(indVarName)) != null) {
            if (bl.getHead() == this) {
                return decl;
            }
        }

        return null;
    }

    // check the canonical form, "for(..., v = lb; v < up; v++)"
    @Override
    public void Canonicalize()
    {
        is_canonical = false;

        Xobject e, ind_var, ub, lb, step;
        Statement s;
        BasicBlock cond = getCondBBlock();

        // check condition part
        if(cond.getHead() != null)
            return;
        if((e = cond.getExpr()) == null)
            return;
        switch(e.Opcode()) {
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
            break;
        default:
            return;
        }
        ind_var = e.left();
        if(!ind_var.isVariable())
            return;
        // pointer type is allowed in OpenMP 3.0
        if(!ind_var.Type().isIntegral() && !ind_var.Type().isPointer())
            return;
        ub = e.right();

        // check init part. the tail must be v=lb
        if (init_part.getHead() == null) {
          // modified for XcalableMP
          Xobject ind_var_decl = 
	    this.getTranslatedForBlockIndVarDecl(ind_var.getName());
          if (ind_var_decl == null) {
            return;
          } else {
            Xobject lb_obj = ind_var_decl.getArg(1);
            if (lb_obj == null) {
              return;
            } else {
              init_part.add(Xcons.Set(ind_var, lb_obj));
              lb = lb_obj;
              ind_var_decl.setArg(1, null);
            }
          }
        } else {
//          e = init_part.getHead().getExpr();
//          if(e.Opcode() == Xcode.COMMA_EXPR) {
//            // expand comma expression
//            s = init_part.getHead();
//            for(XobjArgs a = e.getArgs(); a != null; a = a.nextArgs())
//              s = s.add(a.getArg());
//            init_part.getHead().remove(); // remove original comma expr
//          }
          e = init_part.getTail().getExpr();
          if(e.Opcode() != Xcode.ASSIGN_EXPR)
            return;
          if(!e.left().equals(ind_var))
            return;
          lb = e.right();
        }

        // check iteration expression, and canonicalize
        // only first expression is recognized as iteration expression
        // if(!iter_part.isSingle()) return;

        e = iter_part.getHead().getExpr();
        switch(e.Opcode()) {
        case PRE_INCR_EXPR:
        case POST_INCR_EXPR:
            if(!e.operand().equals(ind_var))
                return;
            step = Xcons.IntConstant(1);
            break;
        case PRE_DECR_EXPR:
        case POST_DECR_EXPR:
            if(!e.operand().equals(ind_var))
                return;
            step = Xcons.IntConstant(-1);
            break;
        case ASG_PLUS_EXPR:
            if(!e.left().equals(ind_var))
                return;
            step = e.right();
            break;
        case ASG_MINUS_EXPR:
            if(!e.left().equals(ind_var))
                return;
            step = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, e.right());
            break;
        case ASSIGN_EXPR:
            if(!e.left().equals(ind_var))
                return;
            switch(e.right().Opcode()) {
            case PLUS_EXPR:
                if(e.right().left().equals(ind_var)) {
                    step = e.right().right();
                } else if(e.right().right().equals(ind_var)) {
                    step = e.right().left();
                } else
                    return;
                break;
            case MINUS_EXPR:
                if(e.right().left().equals(ind_var)) {
                    step = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, e.right().right());
                } else
                    return;
                break;
            default:
                return;
            }
            break;
        default:
            return;
        }
        // canonicalize iteration expression
        if(e.Opcode() != Xcode.ASG_PLUS_EXPR) {
            iter_part.getHead().setExpr(Xcons.asgOp(Xcode.ASG_PLUS_EXPR, ind_var, step));
        }
//        if(!iter_part.isSingle()) {
//            BasicBlock bb = new BasicBlock();
//            while((s = iter_part.getHead().getNext()) != null) {
//                s.remove();
//                bb.add(s);
//            }
//            body.add(bb);
//        }

        // canonicalize conditional expression
        e = cond.getExpr();
        if(e.Opcode() == Xcode.LOG_LE_EXPR) {
            this.min_upperb = ub.copy();
            ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub,/* step */Xcons.IntConstant(1));
            cond.setExpr(Xcons.binaryOp(Xcode.LOG_LT_EXPR, ind_var, ub));
        } else if(e.Opcode() == Xcode.LOG_GE_EXPR) {
            this.min_upperb = ub.copy();
            ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub,/* step */Xcons.IntConstant(-1));
            cond.setExpr(Xcons.binaryOp(Xcode.LOG_GT_EXPR, ind_var, ub));
        }
        else { // Xcode.LOG_LT_EXPR, Xcode.LOG_GT_EXPR
            Xobject ub_copy = ub.copy();
            this.min_upperb = Xcons.binaryOp(Xcode.MINUS_EXPR, ub_copy,/* step */Xcons.IntConstant(1));
        }
        
        this.ind_var = ind_var;
        this.upperb = ub;
        this.lowerb = lb;
        this.step = step;
        is_canonical = true;
    }

    @Override
    public boolean isCanonical()
    {
        return is_canonical;
    }

    @Override
    public Xobject getInductionVar()
    {
        return ind_var;
    }

    @Deprecated
    public Xobject getLoopVar()
    {
        return getInductionVar();
    }

    @Override
    public Xobject getLowerBound()
    {
        return lowerb;
    }

    @Override
    public Xobject getUpperBound()
    {
        return upperb;
    }

    // original function
    public Xobject getMinUpperBound()
    {
        return min_upperb;
    }

    @Override
    public Xobject getStep()
    {
        return step;
    }

    @Override
    public Xcode getCheckOpcode()
    {
        return getCondBBlock().getExpr().Opcode();
    }

    @Override
    public void setLowerBound(Xobject x)
    {
        init_part.getTail().getExpr().setRight(x);
    }

    @Override
    public void setUpperBound(Xobject x)
    {
        BasicBlock cond = getCondBBlock();
        cond.getExpr().setRight(x);
    }

    @Override
    public void setStep(Xobject x)
    {
        iter_part.getHead().getExpr().setRight(x);
    }

    @Override
    public Xobject toXobject()
    {
        Xobject init_x = null, cond_x = null, iter_x = null;
        if(getInitBBlock() != null)
            init_x = getInitBBlock().toXobject();
        if(getCondBBlock() != null)
            cond_x = getCondBBlock().toXobject();
        if(getIterBBlock() != null)
            iter_x = getIterBBlock().toXobject();
        Xobject x = new XobjList(Opcode(), init_x, cond_x, iter_x, getBody().toXobject());
        x.setLineNo(getLineNo());
        return x;
    }
}
