package exc.object;

public class FvarRef extends XobjList
{
  // public FvarRef(Ident id)
  // {
  //   super(Xcode.F_VAR_REF, id.getAddr());
  // }

  public FvarRef(Xobject var)
  {
    super(Xcode.F_VAR_REF, var);
  }

  public FvarRef(Xtype type, Xobject var)
  {
    super(Xcode.F_VAR_REF, type, var);
  }

  public Xobject getVar()
  {
    return this.getArg(0);
  }
}
