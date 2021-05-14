package exc.object;

public class FarrayRef extends XobjList
{

  // arg(0) : F_VAR_REF
  // arg(1) : LIST (subscripts)

  public FarrayRef(Xobject var, XobjList l)
  {
    super(Xcode.F_ARRAY_REF, var.Type().getRef(), var);
    add(l);
  }

  public FvarRef getVarRef()
  {
    return (FvarRef)this.getArg(0);
  }

  public Xobject getVar()
  {
    return this.getArg(0).getArg(0);
  }

  public XobjList getIndices()
  {
    return (XobjList)this.getArg(1);
  }
  
  public Xobject getIndex(int i)
  {
    return this.getArg(1).getArg(i);
  }

  public void setIndex(int i, Xobject x)
  {
    this.getArg(1).setArg(i, x);
  }
}
