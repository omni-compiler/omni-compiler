/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;
import exc.block.Block;

public class XobjString extends XobjConst
{
    String value;

    public XobjString(Xcode code, Xtype type, String value, String fkind)
    {
        super(code, type, fkind);
        this.value = value.intern();
    }

    public XobjString(Xcode code, Xtype type, String value)
    {
        this(code, type, value, null);
    }

    public XobjString(Xcode code, String value)
    {
        this(code, null, value);
    }

    @Override
    public String getString()
    {
        return value;
    }

    @Override
    public String getSym()
    {
        return value;
    }

    // used by xcalablemp package
    public void setSym(String newValue)
    {
        value = newValue;
    }

    @Override
    public String getName()
    {
        return value;
    }
    
    public void setName(String newValue)
    {
        value = newValue;
    }

    @Override
    public Xobject cfold(Block block)
    {
      Ident ident = findIdent(block);
      if (ident == null)
        return this.copy();

      return ident.cfold(block);
    }

    @Override
    public Xobject getSubscripts()
    {
      switch (Opcode()) {
      case VAR:
        return null;
      default:
        break;
      }
      throw new UnsupportedOperationException(toString());
    }

    @Override
    public int getFrank(Block block)
    {
      switch (Opcode()) {
      case VAR:
        Ident ident = findIdent(block);
        if (ident == null)
          throw new UnsupportedOperationException(toString());
        return ident.getFrank();
      default:
        return 0;       // scalar
      }
    }

    public Ident findIdent(Block block)
    {
      if (value == null) 
        return null;

      Xobject id_list = block.getBody().getIdentList();
      if (id_list == null)
        return null;

      Ident ident = id_list.findVarIdent(value);
      return ident;
    }


    public Xobject lbound(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.lbound(dim);
    }
    public Xobject ubound(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.ubound(dim);
    }
    public Xobject extent(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.extent(dim);
    }

    public Xobject[] lbounds(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.lbounds();
    }
    public Xobject[] ubounds(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.ubounds();
    }
    public Xobject[] extents(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.extents();
    }


    @Override
    public Xobject copy()
    {
        return copyTo(new XobjString(code, type, value, getFkind()));
    }

    @Override
    public boolean equals(Xobject x)
    {
        return super.equals(x) && value.equals(x.getString());
    }

    @Override
    public String toString()
    {
        return "(" + OpcodeName() + " " + value + ")";
    }
}
