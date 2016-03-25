/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

/**
 * A block object to represent the function body.
 */

public class FunctionBlock extends CompoundBlock
{
    private Xobject name;
    private Xobject gcc_attrs;
    XobjectDefEnv env;

    /**
     * contructor with function name "name", id list, decls, body block and env.
     * id_list and decls are for parameters.
     */
  public FunctionBlock(Xobject name, Xobject id_list, 
		       Xobject decls, Block body_block,
		       Xobject gcc_attrs, XobjectDefEnv env)
    {
        super(Xcode.FUNCTION_DEFINITION, new BlockList(id_list, decls));
        this.env = env;
        this.name = name;
        this.gcc_attrs = gcc_attrs;
        body.add(body_block);
    }

  public FunctionBlock(Xcode opcode, Xobject name, Xobject id_list, 
		       Xobject decls, Block body_block,
		       Xobject gcc_attrs, XobjectDefEnv env)
    {
        super(opcode, new BlockList(id_list, decls));
        this.env = env;
        this.name = name;
        this.gcc_attrs = gcc_attrs;
        body.add(body_block);
    }

    /** returns Function name */
    public String getName()
    {
        return name.getName();
    }
    
    public Xobject getNameObj()
    {
        return name;
    }

    /** return associated XobjectFile env */
    public XobjectDefEnv getEnv()
    {
        return env;
    }

    /** contert to Xobject */
    @Override
    public Xobject toXobject()
    {
	Xobject x;
	// System.out.println("opcode = "+Opcode()+" head="+body.head);
	// System.out.println("body = "+body);
	// System.out.println("body.toXobject = "+body.toXobject());
	if(Opcode() == Xcode.F_MODULE_DEFINITION){
	    x = new XobjList(Opcode(),
			     name, body.id_list,
			     body.decls, 
			     body.toXobject());
	} else {
	    x = new XobjList(Opcode(),
			     name, body.id_list,
			     body.decls, 
			     (body.head != null)?
			     body.head.toXobject():null, 
			     gcc_attrs);
	}
        x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(FunctionBlock ");
        s.append(name);
        s.append(" ");
        s.append(body);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
