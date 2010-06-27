/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

/**
 * Block object to express If statement.
 */
public class IfBlock extends Block
{
    BlockList then_part;
    BlockList else_part;

    public IfBlock(BasicBlock cond, BlockList then_part, BlockList else_part)
    {
        this(Xcode.IF_STATEMENT, cond, then_part, else_part, null);
    }
    
    /** constructor */
    public IfBlock(Xcode code, BasicBlock cond, BlockList then_part, BlockList else_part, String construct_name)
    {
        super(code, cond, construct_name);
        this.then_part = then_part;
        if(then_part != null)
            then_part.parent = this;
        this.else_part = else_part;
        if(else_part != null)
            else_part.parent = this;
    }
    
    /** constructor to make copy of b */
    public IfBlock(IfBlock b)
    {
        super(b);
        if(b.then_part != null)
            this.then_part = b.then_part.copy();
        if(b.else_part != null)
            this.else_part = b.else_part.copy();
    }

    /** return copy of this block */
    @Override
    public Block copy()
    {
        return new IfBlock(this);
    }

    /** return BasicBlock in condition part */
    @Override
    public BasicBlock getCondBBlock()
    {
        return bblock;
    }

    /** set BasicBlock in condition part */
    public void setCondBBlock(BasicBlock bb)
    {
        bblock = bb;
        bb.parent = this;
    }

    /** get the BlockList of "then" part */
    @Override
    public BlockList getThenBody()
    {
        return then_part;
    }

    /** get the BlockList of "else" part */
    @Override
    public BlockList getElseBody()
    {
        return else_part;
    }

    /** set the BlockList of "then" part */
    @Override
    public void setThenBody(BlockList s)
    {
        then_part = s;
        s.parent = this;
    }

    /** set the BlockList of "else" part */
    @Override
    public void setElseBody(BlockList s)
    {
        else_part = s;
        s.parent = this;
    }

    /** apply visitor recursively */
    @Override
    public void visitBody(BasicBlockVisitor v)
    {
        v.visit(then_part);
        v.visit(else_part);
    }

    /** convert to Xobject */
    @Override
    public Xobject toXobject()
    {
        Xobject then_x = null;
        Xobject else_x = null;
        if(getThenBody() != null)
            then_x = getThenBody().toXobject();
        if(getElseBody() != null)
            else_x = getElseBody().toXobject();
        
        Xobject x;
        
        switch(Opcode()) {
        case IF_STATEMENT:
            x = new XobjList(Opcode(), getCondBBlock().toXobject(), then_x, else_x);
            break;
        case F_IF_STATEMENT:
        case F_WHERE_STATEMENT:
            x = new XobjList(Opcode(), getConstructNameObj(), getCondBBlock().toXobject(), then_x, else_x);
            break;
        default:
            throw new IllegalStateException(Opcode().toString());
        }

        x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(IfBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(then_part);
        s.append(" ");
        s.append(else_part);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
