package exc.block;

//
// inteface to Block.visitBasicBlock()
//
public interface BasicBlockVisitor
{
    public void visit(BasicBlock bb);

    public void visit(Block b);

    public void visit(BlockList b_list);
}
