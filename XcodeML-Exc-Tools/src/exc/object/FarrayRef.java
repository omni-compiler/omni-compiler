package exc.object;

public class FarrayRef extends XobjList
{

  // arg(0) : F_VAR_REF
  // arg(1) : LIST (subscripts)

  public FarrayRef(Xcode code, Xtype type)
  {
    super(code, type);
  }

  public FarrayRef(Xobject var, XobjList l)
  {
    super(Xcode.F_ARRAY_REF, var.Type().getRef(), var);
    add(l);
  }

  public Xobject getVarRef()
  {
    return this.getArg(0);
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

  @Override
  public Xobject copy()
  {
    FarrayRef x = new FarrayRef(code, type);

    for (XobjArgs a = args; a != null; a = a.nextArgs()){
      if (a.getArg() == null)
	x.add(null);
      else
	x.add(a.getArg().copy());
    }

    return copyTo(x);
  }

}
