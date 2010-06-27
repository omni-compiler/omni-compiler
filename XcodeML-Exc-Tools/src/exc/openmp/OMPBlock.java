/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.block.*;
import exc.object.*;

import java.util.List;

/**
 * OpenMP Block
 * 
 * Forall,ParallelRegion,Sections,
 * Single,Ordered,Master,Critical,Atomic,Flush,Barrier
 */
public class OMPBlock extends Block
{
    BasicBlock end_bb;
    BlockList body;
    int narg;
    Xobject info; // note: info is not check in flow analysis.

    public OMPBlock(Xcode code)
    {
        super(code, new BasicBlock());
        end_bb = new BasicBlock();
        end_bb.setParent(this);
    }

    public OMPBlock(Xcode code, BasicBlock bblock)
    {
        super(code, bblock);
        end_bb = new BasicBlock();
        end_bb.setParent(this);
    }

    public OMPBlock(Xcode code, BasicBlock bb1, BasicBlock bb2)
    {
        super(code, bb1);
        end_bb = bb2;
        end_bb.setParent(this);
    }

    /* access function */
    public BasicBlock beginBasicBlock()
    {
        return getBasicBlock();
    }

    public boolean isBeginBasicBlock(BasicBlock b)
    {
        return b == getBasicBlock();
    }

    public BasicBlock endBasicBlock()
    {
        return end_bb;
    }

    public boolean isEndBasicBlock(BasicBlock b)
    {
        return end_bb == b;
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
        s.setParent(this);
    }

    @Override
    public Xobject getInfoExpr()
    {
        return info;
    }

    public int getNarg()
    {
        return narg;
    }

    public void setNarg(int n)
    {
        narg = n;
    }

    @Override
    public void visitBasicBlock(BasicBlockVisitor v)
    {
        v.visit(beginBasicBlock());
        v.visit(endBasicBlock());
    }

    /*
     * static constructor interface
     */
    public static OMPBlock Barrier()
    {
        return new OMPBlock(Xcode.OMP_BARRIER);
    }

    public static OMPBlock Flush(List<Xobject> flush_vars)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_FLUSH);
        if(flush_vars != null) {
            /* expand vector to list */
            Xobject l = Xcons.List();
            for(Xobject v : flush_vars) {
                l.add(v);
            }
            b.info = l;
        }
        return b;
    }

    public static OMPBlock Master(BlockList body)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_MASTER);
        b.setBody(body);
        return b;
    }

    public static OMPBlock Ordered(BlockList body)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_ORDERED);
        b.setBody(body);
        return b;
    }

    public static OMPBlock Single(BlockList body)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_SINGLE);
        b.setBody(body);
        return b;
    }

    public static OMPBlock Sections(int n, BlockList body)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_SECTIONS);
        b.setBody(body);
        b.narg = n;
        return b;
    }

    public static OMPBlock Critical(Xobject name, BlockList body)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_CRITICAL);
        b.setBody(body);
        b.info = name;
        return b;
    }

    public static OMPBlock Atomic(Xobject expr)
    {
        OMPBlock b = new OMPBlock(Xcode.OMP_ATOMIC);
        BlockList body = Bcons.emptyBody();
        body.add(Bcons.Statement(expr));
        b.setBody(body);
        return b;
    }

    public static OMPparallelBlock ParallelRegion(int narg, BasicBlock setup_bb, BlockList body,
        Xobject cond)
    {
        return new OMPparallelBlock(narg, setup_bb, body, cond);
    }

    public static OMPforallBlock Forall(Xobject var, Xobject lb, Xobject ub, Xobject step,
        Xcode checkOpcode, BlockList body, OMPpragma sched, Xobject chunk, boolean ordered)
    {
        return new OMPforallBlock(var, lb, ub, step, checkOpcode, body, sched, chunk, ordered);
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(OMPBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(body);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(" ");
        s.append(end_bb);
        s.append(")");
        return s.toString();
    }
}
