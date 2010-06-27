/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.block.*;
import exc.object.*;

/**
 * OpenMP parallel FOR Block
 *  - loop variable
 *  - lower bound, upper bound, and step (variable?)
 *  - loop scheduling and chunk
 * store in the form (SET var (LIST lb ub step)) at BB expr
 */
public class OMPforallBlock extends OMPBlock
{
    OMPpragma sched;
    Xcode checkOp;
    boolean ordered;
    Statement iter_info;

    public OMPforallBlock(Xobject var, Xobject lb, Xobject ub, Xobject step, Xcode checkOp,
        BlockList body, OMPpragma sched, Xobject chunk, boolean ordered)
    {
        super(Xcode.OMP_FORALL);

        // add leading pad and exit pad
        body.add(new BasicBlock());
        body.insert(new BasicBlock());
        setBody(body);

        this.sched = sched;
        this.ordered = ordered;
        this.checkOp = checkOp;
        Xobject args = Xcons.List(lb, ub, step); // for flow analysis.
        if(chunk != null)
            args.add(chunk);
        iter_info = new Statement(Xcons.Set(var, args));
        this.getBasicBlock().add(iter_info);
    }

    public Xobject getLoopVar()
    {
        return iter_info.getExpr().getArg(0);
    }

    public Xobject getLowerBound()
    {
        return iter_info.getExpr().getArg(1).getArg(0);
    }

    public Xobject getUpperBound()
    {
        return iter_info.getExpr().getArg(1).getArg(1);
    }

    public Xobject getStep()
    {
        return iter_info.getExpr().getArg(1).getArg(2);
    }

    public Xcode getCheckOpcode()
    {
        return checkOp;
    }

    public boolean isOrdered()
    {
        return ordered;
    }

    public OMPpragma getSched()
    {
        return sched;
    }

    public Xobject getChunk()
    {
        XobjArgs args = iter_info.getExpr().getArg(1).getArgs().nextArgs().nextArgs().nextArgs();
        if(args != null)
            return args.getArg();
        else
            return null;
    }
}
