package exc.object;

import static xcodeml.util.XmLog.fatal;

public class MemberRef extends XobjList
{

  // arg(0) : VAR_REF
  // arg(1) : IDENT

  public MemberRef(Xcode code, Xtype type)
  {
    super(code, type);
  }
  
  public MemberRef(Xobject x, String member_name)
  {
    super(Xcode.MEMBER_REF, x.Type(), x);
    assert x.Opcode() == Xcode.F_VAR_REF;

    //Xtype type = x.Type();
    //type = type.getRef(); ??????
    if (!x.Type().isStruct() && !x.Type().isUnion())
      fatal("memberAddr: not struct/union");
        
    Ident mid = type.getMemberList().getMember(member_name); // id_list
    if (mid == null)
      fatal("memberAddr: member name is not found: " + member_name);
        
    this.type = mid.Type();

    add(Xcons.Symbol(Xcode.IDENT, mid.Type(), mid.getName()));
  }

  // public Xobject getVarRef()
  // {
  //   return this.getArg(0);
  // }
  public FvarRef getVarRef()
  {
    return (FvarRef)this.getArg(0);
  }

  public Xobject getVar()
  {
    return this.getArg(0).getArg(0);
  }

  public void setVarRef(Xobject var)
  {
    this.setArg(0, var);
  }
  
  public Xobject getMember()
  {
    return this.getArg(1);
  }

  public String getMemberName()
  {
    return this.getArg(1).getName();
  }

  /* returns the copy of this MemberRef object. */
  @Override
  public Xobject copy()
  {
    MemberRef x = new MemberRef(code, type);
    for (XobjArgs a = args; a != null; a = a.nextArgs()) {
      if (a.getArg() == null)
	x.add(null);
      else
	x.add(a.getArg().copy());
    }
    return copyTo(x);
  }

}
