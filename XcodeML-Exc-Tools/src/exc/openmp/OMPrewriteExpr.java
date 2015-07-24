/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import xcodeml.util.XmOption;
import exc.object.*;
import exc.block.*;

/**
 * pass2: check and write variables
 */
public class OMPrewriteExpr
{
    public OMPrewriteExpr()
    {
    }

    public void run(FuncDefBlock def, OMPfileEnv env)
    {
        OMP.debug("pass2:");
        FunctionBlock fblock = def.getBlock();
        BasicBlockExprIterator bbiter = new BasicBlockExprIterator(fblock);
        for(bbiter.init(); !bbiter.end(); bbiter.next()) {
            Xobject x = bbiter.getExpr();
            if(x != null) {
                bbiter.setExpr(rewriteExpr(
                    x, bbiter.getBasicBlock().getParent(), bbiter.getLineNo(), false));
            }
        }
    }

    public void run(Xobject exprs, Block parent_block, OMPfileEnv env)
    {
        bottomupXobjectIterator itr = new bottomupXobjectIterator(exprs);
        
        for(itr.init(); !itr.end(); itr.next()) {
            Xobject x  = itr.getXobject();
            if(x != null)
                itr.setXobject(rewriteExpr(x, parent_block, x.getLineNo(), false));
        }
    }

    void checkReductionExpr(OMPvar v, Xobject x, Xobject parent)
    {
        x.setProp(OMP.prop, v); /* mark it */
        if(parent == null)
            return;
        switch(parent.Opcode()) {
        case POST_INCR_EXPR:
        case POST_DECR_EXPR:
        case PRE_INCR_EXPR:
        case PRE_DECR_EXPR:
            /* this must be ok */
            parent.setOperand(v.Ref());
            break;
        case ASG_PLUS_EXPR:
        case ASG_MUL_EXPR:
        case ASG_MINUS_EXPR:
        case ASG_BIT_AND_EXPR:
        case ASG_BIT_OR_EXPR:
        case ASG_BIT_XOR_EXPR:
            /* left must be x */
            if(parent.left() == x)
                parent.setLeft(v.Ref());
            break;
        case ASSIGN_EXPR:
            /* left must be x */
            if(parent.left() != x)
                break;
            Xobject rv = parent.right();
            switch(rv.Opcode()) {
            case PLUS_EXPR:
            case MUL_EXPR:
            case BIT_AND_EXPR:
            case BIT_OR_EXPR:
            case BIT_XOR_EXPR:
            case LOG_AND_EXPR:
            case LOG_OR_EXPR:
                if(((OMPvar)rv.right().getProp(OMP.prop)) == v) {
                    parent.setLeft(v.Ref());
                    rv.setRight(v.Ref());
                    break;
                }
            case MINUS_EXPR:
                if(((OMPvar)rv.left().getProp(OMP.prop)) == v) {
                    parent.setLeft(v.Ref());
                    rv.setLeft(v.Ref());
                    break;
                }
            }
        }
    }

    // pass2:
    // check and rewrite variable reference
    private Xobject rewriteExpr(Xobject x, Block b, LineNo ln, boolean inType)
    {
        if(x == null)
            return null;

        OMPvar v;
        XobjectIterator i = new bottomupXobjectIterator(x);
        for(i.init(); !i.end(); i.next()) {
            Xobject xx = i.getXobject();
            OMP.debug("expr1=" + xx);
            if(xx == null)
                continue;

            if(xx instanceof Ident) {
                if((v = OMPinfo.findOMPvarBySharedOrPrivate(b, xx.getName())) == null)
                    continue;
                if(XmOption.isLanguageF())
                    rewriteType(xx, b);
            } else if(xx.isVariable()) {
                if((v = OMPinfo.refOMPvar(b, xx.getName())) == null)
                    continue;
                if(v.is_reduction)
                    checkReductionExpr(v, xx, i.getParent());
                else {
                    i.setXobject(inType ? v.SharedRef() : v.Ref());
                    if(XmOption.isLanguageF())
                        rewriteType(v, b);
                }
            } else if(xx.isVarAddr() || xx.isArray() || xx.isArrayAddr()) {
                if((v = OMPinfo.refOMPvar(b, xx.getName())) == null)
                    continue;
                if(v != null)
                    i.setXobject(inType ? v.getSharedAddr() : v.getAddr());
                if(v.is_reduction) {
                    OMP.warning(ln,
                        "may be bad reference for reduction variable,'" + xx.getName() + "'");
                }
            } else if(xx.Opcode() == Xcode.ASSIGN_EXPR
                && (v = ((OMPvar)xx.left().getProp(OMP.prop))) != null) {
                checkReductionExpr(v, xx.left(), xx);
            }
        }

        /* check remaining reduction variable */
        for(i.init(); !i.end(); i.next()) {
            Xobject xx = i.getXobject();
            OMP.debug("expr2=" + xx);
            if(xx == null)
                continue;
            if(xx.isVariable() && (v = ((OMPvar)xx.getProp(OMP.prop))) != null) {
                if(i.getParent() != null && i.getParent().Opcode() == Xcode.ASSIGN_EXPR
                    && i.getParent().left() == xx
                    && v.reduction_op != OMPpragma.DATA_REDUCTION_MIN
                    && v.reduction_op != OMPpragma.DATA_REDUCTION_MAX)
                    OMP.warning(ln,
                        "may be bad reference for reduction variable,'" + xx.getName() + "'");
                i.setXobject(v.Ref());
            }
        }

        return i.topXobject();
    }
    
    private void rewriteType(OMPvar v, Block b)
    {
        rewriteType(v.getSharedAddr(), b);
        rewriteType(v.getPrivateAddr(), b);
    }
    
    private void rewriteType(Xobject e, Block b)
    {
        if(e == null)
            return;
        
        Xtype t = e.Type();
        
        if(t == null)
            return;
        
        Xobject x;
        
        switch(t.getKind()) {
        case Xtype.BASIC:
            t = t.copy();
            e.setType(t);
            if(XmOption.isLanguageF()) {
                x = rewriteExpr(t.getFlen(), b, b.getLineNo(), true);
                ((BasicType)t).setFlen(x);
            }
            break;
        case Xtype.ARRAY:
            t = t.copy();
            e.setType(t);
            x = rewriteExpr(t.getArraySizeExpr(), b, b.getLineNo(), true);
            ((ArrayType)t).setArraySizeExpr(x);
            break;
        case Xtype.F_ARRAY: {
            t = t.copy();
            e.setType(t);
            Xobject[] exprs = t.getFarraySizeExpr();
            for(int i = 0; i < exprs.length; ++i) {
                x = rewriteExpr(exprs[i].copy(), b, b.getLineNo(), true);
                exprs[i] = x;
            }
        }
            break;
        }
    }
}
