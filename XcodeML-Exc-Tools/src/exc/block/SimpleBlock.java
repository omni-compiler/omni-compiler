/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

//
// Simple Block, which has only basic block
//
public class SimpleBlock extends Block
{

    public SimpleBlock(Xcode code, BasicBlock bblock)
    {
        super(code, bblock);
    }

    public SimpleBlock(Xcode code)
    {
        this(code, new BasicBlock());
    }

    public SimpleBlock(SimpleBlock b)
    {
        super(b);
    }
    
    @Override
    public Block copy()
    {
        return new SimpleBlock(this);
    }

    @Override
    public Xobject toXobject()
    {
        Xobject x = new XobjList(code, bblock.toXobject());
        x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(SimpleBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
