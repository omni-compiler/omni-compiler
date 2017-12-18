package exc.block;

import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjString;
import exc.object.Xobject;
import exc.object.XobjectDef;
import exc.object.XobjectFile;

public class FmoduleBlock extends CompoundBlock
{
    private Xobject name;
    private Xobject parent_name;
    
    private BlockList func_blocks;
    
    private XobjectFile env;

    public FmoduleBlock(String name, XobjectFile env)
    {
      this(name, null, env);
    }

    public FmoduleBlock(String name, String parent_name, XobjectFile env)
    {
        super(Xcode.F_MODULE_DEFINITION, new BlockList(Xcons.IDList(), Xcons.List()));
        this.name = new XobjString(Xcode.IDENT, name);
        this.parent_name = (parent_name != null) ? new XobjString(Xcode.IDENT, parent_name) : null;
        this.env = env;
        this.func_blocks = new BlockList();
    }
    
    public void addFunctionBlock(FunctionBlock func_block)
    {
        func_blocks.add(func_block);
    }
    
    public Xobject getName()
    {
        return name;
    }

    public Xobject getParentName()
    {
        return parent_name;
    }
    
    public boolean isSub()
    {
        return parent_name != null;
    }

    public BlockList getFunctionBlocks()
    {
        return func_blocks;
    }
    
    public FunctionBlock getFunctionBlock(String name)
    {
        for(Block b = func_blocks.getHead(); b != null; b = b.getNext()) {
            if(((FunctionBlock)b).getName().equals(name))
                return (FunctionBlock)b;
        }
        
        return null;
    }
    
    public boolean hasVar(String name)
    {
        Xobject idList = getBody().getIdentList();
        return (idList.findVarIdent(name) != null);
    }
    
    public XobjectDef toXobjectDef()
    {
        Xobject xcontains = Xcons.List(Xcode.F_CONTAINS_STATEMENT);
        for(Block b = func_blocks.getHead(); b != null; b = b.next) {
            xcontains.add(b.toXobject());
        }
        
        return new XobjectDef(Xcons.List(Xcode.F_MODULE_DEFINITION,
            name, getBody().getIdentList(), getBody().getDecls(), xcontains, parent_name), env);
    }
}
