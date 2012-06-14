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

  /* 
   * Method to analyze and handle nodes directives.
   *  nodeDecl = (nodeMaptype name dimensions inheritDecl)
   *     - nodeMaptype is not used
   */
  public static void analyzePragma(Xobject nodesDecl,XMPenv env, PragmaBlock pb) {
    XMPnodes nodesObject = new XMPnodes();
    nodesObject.parsePragma(nodesDecl,env,pb);
    env.putXMPobject(nodesObject,pb);
    if(XMP.debugFlag){
      System.out.println("nodesObject="+nodesObject);
    }
  }
    
  /* parse nodesDecl and set info to this object */
  public void parsePragma(Xobject decl, XMPenv env, PragmaBlock pb) {

    // check name collision, name = arg(1)
    _name = decl.getArg(0).getString();
    env.checkObjectNameCollision(_name,pb);
    if(XMP.hasError()) return;

    // declare nodes desciptor
    _descId = env.declObjectId(XMP.DESC_PREFIX_ + _name, pb);

    // declare nodes object, count the numer of Dimension, demension = arg(2)
    _dim = 0;
    for (XobjArgs i = decl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      _sizeVector.add(XMPdimInfo.parseDecl(i.getArg()));
      _dim++;
    }
    if (_dim > XMP.MAX_DIM){
      XMP.error("nodes dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    boolean isDynamic = false;
    for(int k = 0; k < _dim; k++){
      if(_sizeVector.elementAt(k).isStar()){
	if(k == (_dim-1))
	  isDynamic = true;
	else {
	  XMP.error("* must be only in the last dimension in nodes");
	  return;
	}
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
  public Block buildConstructor(XobjectDef def){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f = def.declExternIdent(XMP.nodes_alloc_f,Xtype.FsubroutineType);
    Xobject args = Xcons.List(_descId.Ref(),Xcons.IntConstant(_dim));
    bb.add(f.callSubroutine(args));

    f = def.declExternIdent(XMP.nodes_dim_size_f,Xtype.FsubroutineType);
    for(int i = 0; i < _dim; i++){
      args = Xcons.List(_descId.Ref(),Xcons.IntConstant(i),
 			_sizeVector.elementAt(i).getSize());
       bb.add(f.callSubroutine(args));
     }

    switch(inheritType){
    case INHERIT_GLOBAL:
      f = def.declExternIdent(XMP.nodes_init_GLOBAL_f,Xtype.FsubroutineType);
      bb.add(f.callSubroutine(Xcons.List(_descId.Ref())));
      break;
    case INHERIT_EXEC:
      f = def.declExternIdent(XMP.nodes_init_EXEC_f,Xtype.FsubroutineType);
      bb.add(f.callSubroutine(Xcons.List(_descId.Ref())));
      break;
    case INHERIT_NODES:
      // create nodeRef ...
    default:
      XMP.fatal("bulidConstrutor: unknown inheritType="+inheritType);
    }
    return b;
  }
}
