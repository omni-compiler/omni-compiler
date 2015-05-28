/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.block.*;
import exc.object.*;

import java.util.Vector;

/* 
 * XMP Nodes object 
 */
public class XMPnodes extends XMPobject {
  public final static int INHERIT_GLOBAL	= 10;
  public final static int INHERIT_EXEC		= 11;
  public final static int INHERIT_NODES		= 12;

  int inheritType;
  XMPobjectsRef nodesRef;  // in case of INHERIT_NODES

  boolean isDynamic = false;
  private Vector<XMPdimInfo> _sizeVector;
  
  // null constructor
  public XMPnodes() { 
    super(XMPobject.NODES);
    _sizeVector = new Vector<XMPdimInfo>();

  }

  public String toString(){
    String s = "{Nodes:";
    switch(inheritType){
    case INHERIT_GLOBAL:    
      s += "(G)"; break;
    case INHERIT_EXEC:
      s += "(E)"; break;
    case INHERIT_NODES:
      s += "(N)"; break;
    default:
      s += "(?)"; break;
    }
    s += _sizeVector;
    s += "}";
    return s;
  }

  public XMPdimInfo getInfoAt(int index) {
    return _sizeVector.get(index);
  }

  /* 
   * Method to analyze and handle nodes directives.
   *  nodeDecl = (nodeMaptype name dimensions inheritDecl)
   *     - nodeMaptype is not used
   */
  public static void analyzePragma(Xobject nodesDecl,XMPenv env, PragmaBlock pb) {
    XMPnodes nodesObject = new XMPnodes();
    nodesObject.parsePragma(nodesDecl,env,pb);
    env.declXMPobject(nodesObject,pb);
    if(XMP.debugFlag){
      System.out.println("nodesObject="+nodesObject);
    }
  }
    
  /* parse nodesDecl and set info to this object */
  public void parsePragma(Xobject decl, XMPenv env, PragmaBlock pb) {

    // check name collision, name = arg(1)
    _name = decl.getArg(0).getString();
    if(env.findXMPobject(_name,pb) != null){
      XMP.errorAt(pb,"XMP object '"+_name+"' is already declared");
      return;
    }

    // declare nodes desciptor
    _descId = env.declObjectId(XMP.DESC_PREFIX_ + _name, pb);

    // declare nodes object, count the numer of Dimension, demension = arg(2)
    _dim = 0;
    for (XobjArgs i = decl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      _sizeVector.add(XMPdimInfo.parseDecl(i.getArg()));
      _dim++;
    }
    if (_dim > XMP.MAX_DIM){
      XMP.errorAt(pb,"nodes dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    boolean isDynamic = false;
    for(int k = 0; k < _dim; k++){
      if(_sizeVector.elementAt(k).isStar()){
	isDynamic = true;
	break;
	// if(k == (_dim-1))
	//   isDynamic = true;
	// else {
	//   XMP.errorAt(pb,"* must be only in the last dimension in nodes");
	//   return;
	// }
      }
    }

    // inhrit type = arg(3)
    XobjList inheritDecl = (XobjList)decl.getArg(2);
    if(inheritDecl == null)
      inheritType = XMPnodes.INHERIT_GLOBAL; //default
    else {
      inheritType = inheritDecl.getArg(0).getInt();
      switch (inheritType){
      case XMPnodes.INHERIT_GLOBAL:
      case XMPnodes.INHERIT_EXEC:
        break;
      case XMPnodes.INHERIT_NODES:
	nodesRef = XMPobjectsRef.parseDecl(inheritDecl.getArg(1), env, pb);
	break;
      default:
	XMP.fatal("unknown nodes inheri type");
      }
    }
  }

  /* code for ndoes directive:
   *    _xmpf_nodes_alloc__(n_desc,#dim)
   *    _xmpf_nodes_dim_size__(n_desc,i_dim,size)
   *
   *    _xmpf_nodes_init_GLOBAL__(n_1)
   *    _xmpf_nodes_init_EXEC__(n_1)
   *    _xmpf_nodes_init_NODES__(n_1,nodes_ref)
   */
  public void buildConstructor(BlockList body, XMPenv env){

    BlockList b;
    if (_is_saveDesc && !env.currentDefIsModule()){
      b = Bcons.emptyBody();
    }
    else {
      b = body;
    }

    Ident flagVar = null;
    if (_is_saveDesc && !env.currentDefIsModule()){

      Xtype save_desc = _descId.Type().copy();
      save_desc.setIsFsave(true);
      _descId.setType(save_desc);

      Xtype save_logical = Xtype.FlogicalType.copy();
      save_logical.setIsFsave(true);
      BlockList bl = env.getCurrentDef().getBlock().getBody();
      flagVar = bl.declLocalIdent(XMP.SAVE_DESC_PREFIX_ + _name, save_logical,
      				    StorageClass.FSAVE,
      				    Xcons.List(Xcode.F_VALUE, Xcons.FlogicalConstant(false)));
    }

    Ident f = env.declInternIdent(XMP.nodes_alloc_f,Xtype.FsubroutineType);
    Xobject args = Xcons.List(_descId.Ref(),Xcons.IntConstant(_dim));
    b.add(f.callSubroutine(args));

    f = env.declInternIdent(XMP.nodes_dim_size_f,Xtype.FsubroutineType);
    for(int i = 0; i < _dim; i++){
      Xobject size = _sizeVector.elementAt(i).getSize();
      if(size == null) size = Xcons.IntConstant(-1);
      args = Xcons.List(_descId.Ref(),Xcons.IntConstant(i), size);
      b.add(f.callSubroutine(args));
    }

    switch(inheritType){
    case INHERIT_GLOBAL:
      f = env.declInternIdent(XMP.nodes_init_GLOBAL_f,Xtype.FsubroutineType);
      b.add(f.callSubroutine(Xcons.List(_descId.Ref())));
      break;
    case INHERIT_EXEC:
      f = env.declInternIdent(XMP.nodes_init_EXEC_f,Xtype.FsubroutineType);
      b.add(f.callSubroutine(Xcons.List(_descId.Ref())));
      break;
    case INHERIT_NODES:
      b.add(nodesRef.buildConstructor(env));
      f = env.declInternIdent(XMP.nodes_init_NODES_f, Xtype.FsubroutineType);
      b.add(f.callSubroutine(Xcons.List(_descId.Ref(), nodesRef.getDescId().Ref())));
      break;
    default:
      XMP.fatal("bulidConstrutor: unknown inheritType="+inheritType);
    }

    if (_is_saveDesc && !env.currentDefIsModule()){
      b.add(Xcons.Set(flagVar.Ref(), Xcons.FlogicalConstant(true)));
      body.add(Bcons.IF(BasicBlock.Cond(Xcons.unaryOp(Xcode.LOG_NOT_EXPR, flagVar.Ref())), b, null));
    }

  }

  public void buildDestructor(BlockList body, XMPenv env){
    if (!_is_saveDesc){
      Ident f = env.declInternIdent(XMP.nodes_dealloc_f,Xtype.FsubroutineType);
      Xobject args = Xcons.List(_descId.Ref());
      body.add(f.callSubroutine(args));
    }
  }
}
