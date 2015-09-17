/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import java.util.ArrayList;
import java.util.List;

import xcodeml.util.XmOption;
import exc.object.*;
import exc.xcodeml.XcodeMLtools;
import exc.block.*;

/**
 * process analyze OpenMP pragma
 * pass1: check directive and allocate descriptor
 */
public class OMPanalyzePragma
{
    FuncDefBlock def;
    OMPfileEnv omp_env;

    public void run(FuncDefBlock def, OMPfileEnv omp_env)
    {
        this.def = def;
        this.omp_env = omp_env;
        Block b;
        Block fblock = def.getBlock();
        OMP.debug("run");
        // pass1: traverse to collect information about OMP pramga
        if(OMP.debugFlag)
            System.out.println("pass1:");
        b = fblock.getBody().getHead();
        b.setProp(OMP.prop, new OMPinfo(OMPpragma.FUNCTION_BODY, null, b, omp_env));

        BlockIterator i = new topdownBlockIterator(fblock);
        for(i.init(); !i.end(); i.next()) {
            b = i.getBlock();
            if(OMP.debugFlag)
                System.out.println("pass1=" + b);
            if(b.Opcode() == Xcode.OMP_PRAGMA)
                checkPragma((PragmaBlock)b);
        }
    }

    OMPinfo outerOMPinfo(Block b)
    {
        OMP.debug("outerOMPinfo");
        for(Block bp = b.getParentBlock(); bp != null; bp = bp.getParentBlock()) {
            if(bp.Opcode() == Xcode.OMP_PRAGMA) {
                return (OMPinfo)bp.getProp(OMP.prop);
            }
        }
        return null;
    }

    void checkPragma(PragmaBlock pb)
    {
        OMPinfo outer = outerOMPinfo(pb);
        OMPinfo info = new OMPinfo(OMPpragma.valueOf(pb.getPragma()), outer, pb, omp_env);
        pb.setProp(OMP.prop, info);
        OMP.debug("checkPragma");

        OMPpragma p = info.pragma;
        OMPpragma c;
        Xobject t;
       Block b;
        List<XobjList> idLists;
        topdownBlockIterator bitr;

        switch(p) {
        case TASK:
    	info.setIfExpr(Xcons.FlogicalConstant(true));
        info.setFinalExpr(Xcons.FlogicalConstant(false));
        case PARALLEL: /* new parallel section */
        case FOR: /* loop <clause_list> */
        case SECTIONS: /* sections <clause_list> */
        	// Assume that the kinds of the cluases are already checked
            idLists = new ArrayList<XobjList>();
            for(Xobject a : (XobjList)pb.getClauses()) {
                c = OMPpragma.valueOf(a.getArg(0));
                switch(c) {
                case DIR_IF:
                    info.setIfExpr(a.getArg(1));
                    break;
                case DATA_FINAL:
                    info.setFinalExpr(a.getArg(1));
                    break;
                case DIR_NOWAIT:
                    info.no_wait = true;
                    break;
                case DIR_ORDERED:
                    info.ordered = true;
                    break;
                case DIR_SCHEDULE:
                    info.setSchedule(a.getArg(1));
                    break;
                case DATA_DEFAULT:
                    info.data_default = OMPpragma.valueOf(a.getArg(1));
                    break;
                case DIR_NUM_THREADS:
                    info.num_threads = a.getArg(1);
                    break;
                case DIR_UNTIED:
                    info.untied = true;
                    OMP.debug("UNTIED");
                    break;
                case DIR_MERGEABLE:
                    info.mergeable = true;
                    OMP.debug("MERGEABLE");
                    break;
                case DATA_DEPEND_IN:
                case DATA_DEPEND_OUT:
                case DATA_DEPEND_INOUT:
                    info.mergeable = true;
                    OMP.debug("DEPEND");
                    break;
                default: // DATA_*
                    for(Xobject aa : (XobjList)a.getArg(1))
                        info.declOMPvar(aa.getName(), c);
                    idLists.add((XobjList)a.getArg(1));
                    break;
                }
            }
            for(XobjList l : idLists)
                info.declOMPVarsInType(l);
            
            if(p == OMPpragma.FOR) {
                if(outer != null && outer.pragma != OMPpragma.PARALLEL) {
                    OMP.error(pb.getLineNo(), "'FOR/DO' directive is nested");
                    return;
                }
                /* check indiction variable */
                if(XmOption.isLanguageC()) {
                    if(pb.getBody().getHead().Opcode() != Xcode.FOR_STATEMENT) {
                        OMP.error(pb.getLineNo(), "FOR loop must follows FOR directive");
                        return;
                    }
                } else {
                    if(pb.getBody().getHead().Opcode() == Xcode.OMP_PRAGMA){
                    	PragmaBlock pb2=(PragmaBlock)pb.getBody().getHead();
                    	OMPinfo outer2 =outerOMPinfo(pb2);
                    	OMPinfo info2 = new OMPinfo(OMPpragma.valueOf(pb2.getPragma()), outer2, pb2, omp_env);
                    	OMPpragma pc=info2.pragma;
                    	switch(pc)
                    	{
                    	case SIMD:
                    		info.simd=true;
                    		break;
                    	}
                    }else
                    if(pb.getBody().getHead().Opcode() != Xcode.F_DO_STATEMENT) {
                        OMP.error(pb.getLineNo(), "DO loop must follows DO directive");
                        return;
                    }	
                }
                if(!info.simd)
                {
                	ForBlock for_block = (ForBlock)pb.getBody().getHead();
                	for_block.Canonicalize();
                	if(!for_block.isCanonical()) {
                		OMP.error(pb.getLineNo(), "not cannonical FOR/DO loop");
                		return;
                	}	
                	
                	Xobject ind_var = for_block.getInductionVar();
                	if(!info.isPrivateOMPvar(ind_var.getName())) {
                		OMPvar v = info.findOMPvar(ind_var.getName());
                		if(v == null) {
                			info.declOMPvar(ind_var.getName(), OMPpragma.DATA_PRIVATE);
                		} else if(!v.is_last_private && v.is_shared) {
                			/* loop variable may be lastprivate */
                			OMP.error(pb.getLineNo(), "FOR/DO loop variable '" + v.id.getName()
                					+ "' is declared as shared");
                			return;
                		}	
                	}	
                }
            }
            if(p == OMPpragma.SECTIONS && outer != null && outer.pragma != OMPpragma.PARALLEL)
                OMP.error(pb.getLineNo(), "'sections' directive is nested");

            bitr = new topdownBlockIterator(pb.getBody());
            for(bitr.init(); !bitr.end(); bitr.next()) {
                if(bitr.getBlock() instanceof FdoBlock) {
                    Xobject x = ((FdoBlock)bitr.getBlock()).getInductionVar();
                    Ident id = pb.findVarIdent(x.getName());
                    if(id != null)
                        id.setIsInductionVar(true);
                }
            }
            break;
        case SIMD:
        case DECLARE:
	    break;	
        case SINGLE: /* single <clause list> */
            for(XobjArgs a = pb.getClauses().getArgs(); a != null; a = a.nextArgs()) {
                t = a.getArg();
                c = OMPpragma.valueOf(t.getArg(0));
                switch(c) {
                case DIR_NOWAIT:
                    info.no_wait = true;
                    break;
                default: // DATA_*
                    for(XobjArgs aa = t.getArg(1).getArgs(); aa != null; aa = aa.nextArgs())
                        info.declOMPvar(aa.getArg().getName(), c);
                }
            }
            if(outer != null &&
                outer.pragma != OMPpragma.PARALLEL &&
                outer.pragma != OMPpragma.PARALLEL_FOR &&
                outer.pragma != OMPpragma.FOR)
                OMP.error(pb.getLineNo(), "'single' directive is nested or is not in parallel region");
            break;

        case MASTER: /* master */
            if(info.findContext(OMPpragma.FOR) != null)
                OMP.error(pb.getLineNo(), "'master' not permitted in the extent of 'for'");
            if(info.findContext(OMPpragma.SECTIONS) != null)
                OMP.error(pb.getLineNo(), "'master' not permitted in the extent of 'sections'");
            if(info.findContext(OMPpragma.SINGLE) != null)
                OMP.error(pb.getLineNo(), "'master' not permitted in the extent of 'single'");
            break;

        case CRITICAL: /* critical <name> */
            t = pb.getClauses();
            if(t != null)
                info.arg = t.getArg(0);
            else
                info.arg = null;
            if(info.findContext(OMPpragma.CRITICAL, info.arg) != null)
                OMP.error(pb.getLineNo(), "'critical' with the same name is nested");
            break;

        case BARRIER: /* barrier */
            if(outer != null && outer.pragma != OMPpragma.PARALLEL)
                OMP.error(pb.getLineNo(), "'barrier' in bad context");
            break;

        case FLUSH: /* flush <namelist> */
            info.setFlushVars(pb.getClauses());
            break;

        case ATOMIC: /* atomic */
            if(pb.getBody().isSingle() &&
                ((b = pb.getBody().getHead()).Opcode() == Xcode.LIST ||
                b.Opcode() == Xcode.F_STATEMENT_LIST)) {
                
                BasicBlock bb = pb.getBody().getHead().getBasicBlock();
                if(bb.isSingle()) {
                    Xobject x = bb.getHead().getExpr();
                    if(isAtomicExpr(x))
                        break;
                    if(x.isSet()) {
                        if((x = makeAtomicExpr(x)) != null) {
                            bb.getHead().setExpr(x);
                            break;
                        }
                    }
                }
            }
            OMP.error(pb.getLineNo(), "bad expression in 'atomic' directive");
            break;

        case ORDERED: /* ordered */
            if(info.findContext(OMPpragma.CRITICAL) != null)
                OMP.error(pb.getLineNo(), "'ordered' not permitted in the extent of 'critical'");
            break;
        case THREADPRIVATE:
            omp_env.declThreadPrivate(def.getDef().getDef(), pb, pb.getClauses());
            break;
        default:
            OMP.fatal("unknown OpenMP pramga = " + p);
            break;
        }
    }

    boolean isAtomicExpr(Xobject x)
    {
        OMP.debug("isAtomicExpr");
        switch(x.Opcode()) {
        case POST_INCR_EXPR:
        case POST_DECR_EXPR:
        case PRE_INCR_EXPR:
        case PRE_DECR_EXPR:
        case ASG_PLUS_EXPR:
        case ASG_MUL_EXPR:
        case ASG_MINUS_EXPR:
        case ASG_DIV_EXPR:
        case ASG_BIT_AND_EXPR:
        case ASG_BIT_OR_EXPR:
        case ASG_BIT_XOR_EXPR:
        case ASG_LSHIFT_EXPR:
        case ASG_RSHIFT_EXPR:
            return true;
        }
        return false;
    }

    /** create atomic expression */
    private Xobject makeAtomicExpr(Xobject x)
    {
        OMP.debug("makeAtomicExpr");
        return XmOption.isLanguageC() ? makeAtomicExprC(x) : makeAtomicExprF(x);
    }
    
    /** C: create atomic expression */
    private Xobject makeAtomicExprC(Xobject x)
    {
        Xobject lv = x.left();
        Xobject rv = x.right();
        OMP.debug("makeAtomicExprC");
        switch(rv.Opcode()) {
        case PLUS_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_PLUS_EXPR, lv, rv.right());
            if(lv.equals(rv.right()))
                return Xcons.asgOp(Xcode.ASG_PLUS_EXPR, lv, rv.left());
            break;
        case MUL_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_MUL_EXPR, lv, rv.right());
            if(lv.equals(rv.right()))
                return Xcons.asgOp(Xcode.ASG_MUL_EXPR, lv, rv.left());
            break;
        case MINUS_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_MINUS_EXPR, lv, rv.right());
            break;
        case DIV_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_DIV_EXPR, lv, rv.right());
            break;
        case BIT_AND_EXPR:
        case LOG_AND_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_BIT_AND_EXPR, lv, rv.right());
            if(lv.equals(rv.right()))
                return Xcons.asgOp(Xcode.ASG_BIT_AND_EXPR, lv, rv.left());
            break;
        case BIT_OR_EXPR:
        case LOG_OR_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_BIT_OR_EXPR, lv, rv.right());
            if(lv.equals(rv.right()))
                return Xcons.asgOp(Xcode.ASG_BIT_OR_EXPR, lv, rv.left());
            break;
        case BIT_XOR_EXPR:
            if(lv.equals(rv.left()))
                return Xcons.asgOp(Xcode.ASG_BIT_XOR_EXPR, lv, rv.right());
            if(lv.equals(rv.right()))
                return Xcons.asgOp(Xcode.ASG_BIT_XOR_EXPR, lv, rv.left());
            break;
        }
        return null;
    }
    
    /** Fortran: create atomic expression */
    private Xobject makeAtomicExprF(Xobject x)
    {
        OMP.debug("makeAtomicExprF");

        if(x.Opcode() != Xcode.F_ASSIGN_STATEMENT)
            return null;
        
        Xobject lv = x.left();
        Xobject rv = x.right();
        switch(rv.Opcode()) {
        case PLUS_EXPR:
        case MINUS_EXPR:
        case DIV_EXPR:
        case MUL_EXPR:
        case F_POWER_EXPR:
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
        case F_LOG_EQV_EXPR:
        case F_LOG_NEQV_EXPR:
        case LOG_AND_EXPR:
        case LOG_OR_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case F_USER_BINARY_EXPR:
            if(lv.equals(rv.left()) || lv.equals(rv.right())) {
                return Xcons.List(rv.Opcode(), lv, rv.left(), rv.right());
            }
            break;
        case FUNCTION_CALL: {
                Xobject args = rv.getArgOrNull(1);
                if(args == null || args.Nargs() != 2)
                    break;
                String funcName = rv.getArg(0).getName();
                if(!funcName.equals("min") && !funcName.equals("max") &&
                    !funcName.equals("iand") && !funcName.equals("ior") &&
                    !funcName.equals("ieor")) {
                    break;
                }
                if(lv.equals(args.left()) || lv.equals(args.right())) {
                    return Xcons.List(rv.Opcode(), lv,
                        args.left(), args.right(), rv.getArg(0));
                }
                break;
            }
        }
        return null;
    }
}
