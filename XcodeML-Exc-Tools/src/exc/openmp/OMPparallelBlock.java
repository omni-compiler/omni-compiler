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
 * OpenMP parallel block
 */
public class OMPparallelBlock extends OMPBlock
{
    public OMPparallelBlock(int narg, BasicBlock setup, BlockList body, Xobject cond)
    {
        super(Xcode.OMP_PARALLEL, setup);
        setNarg(narg);
        setup.setExpr(cond); // set cond exp at end of setup bb.
        setBody(body);
    }

    public BasicBlock setupBasicBlock()
    {
        return getBasicBlock();
    }

    public Xobject getIfExpr()
    {
        return beginBasicBlock().getExpr();
    }
}
