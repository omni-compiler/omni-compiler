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
 * XMP ObjectsRef object 
 */
public class XMPobjectsRef {

  Ident descId;
  
  String refName;
  XMPobject refObject;	// nodes or template

  Vector<XMPdimInfo> subscripts;
  Vector<XMPdimInfo> loop_dims;  // back pointer to loop_dims

  /* 
   * !$XMP loop (i_0,i_1,..,i_k,...) on t(j_0,j_1,...,j_k,...)
   *       j_k == i_l+off then:
   *   subscripts maps tmpl to loop_idx 
   *     subscripts k_th element contains l (getLoopOnIndex)
   *                                    and off (getOnRefOffset)
   *   loop_dims maps loop_idx to tmpl
   *     loop_dims l_th element contains k (getLoopOnIndex)
   */

  public XMPobjectsRef() {} // null constructor

  public Ident getDescId() { return descId; }

  public XMPtemplate getTemplate() { return (XMPtemplate)refObject; }

  public XMPnodes getNodes() { return (XMPnodes)refObject; }

  /* 
   * Nodes Reference:
   *  nodes_ref := (nodes-subscript) | nodes_anme(nodes-subscript, ...)
   *    nodes-subsript = int-expr| triplet| *
   *
   *   (LIST null (LIST lower upper stride))
   *   (LIST name (LIST (LIST lower upper stride) ...))
   * 
   * or Template Ref:
   */
  
  public static XMPobjectsRef parseDecl(Xobject decl,XMPenv env, 
					PragmaBlock pb){
    if(decl == null) return null;
    XMPobjectsRef objRef = new XMPobjectsRef();
    objRef.parse(decl,env,pb);
    return objRef;
  }

  public XMPobject getRefObject() { return refObject; }

  public Vector<XMPdimInfo> getSubscripts() { return subscripts; }

  public int getLoopOnIndex(int i){
    return loop_dims.elementAt(i).getLoopOnIndex();
  }

  public int getOnRefLoopIndex(int i){
    return subscripts.elementAt(i).getOnRefLoopIndex();
  }

  public Xobject getLoopOffset(int i){
    return subscripts.elementAt(i).getOnRefOffset();
  }

  void parse(Xobject decl, XMPenv env, PragmaBlock pb) {
    if (decl.getArg(0) == null) {
      subscripts = new Vector<XMPdimInfo>();
      subscripts.add(XMPdimInfo.parseDecl(decl.getArg(1)));
      refName = "xmp_";
    } else {
      refName = decl.getArg(0).getString();
      refObject = env.findXMPobject(refName,pb);
      if (refObject == null) {
	XMP.error("cannot find objects '" + refName + "'");
	return;
      }
      Xobject subscriptList = decl.getArg(1);
      if(subscriptList != null){
	subscripts = XMPdimInfo.parseSubscripts(subscriptList);
      }
    }
    // allocate DescId
    descId = env.declObjectId(XMP.genSym(refName), pb);
  }

  public void setLoopDimInfo(Vector<XMPdimInfo> dims) { loop_dims = dims;}

  // make contructor
  /*
   * ref_tmpl_alloc_f(ref_id,temp_id,#n_dim)
   *    or ref_node_alloc_f(ref_id, node_id,#n_dim)
   * ref_set_dim_info(ref_id,#dim_i,lb,ub,step)
   * ref_init(ref_id)
   */
  public Block buildConstructor(XMPenv env){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f;

    switch(refObject.getKind()){
    case XMPobject.NODES:
      f = env.declExternIdent(XMP.ref_nodes_alloc_f,Xtype.FsubroutineType);
      break;
    case XMPobject.TEMPLATE:
      f = env.declExternIdent(XMP.ref_templ_alloc_f,Xtype.FsubroutineType);
      break;
    default:
      XMP.fatal("bad object for ref");
      return null;
    }
    Xobject args = Xcons.List(descId.Ref(),refObject.getDescId().Ref(),
			      Xcons.IntConstant(subscripts.size()));
    bb.add(f.callSubroutine(args));
    f = env.declExternIdent(XMP.ref_set_dim_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < subscripts.size(); i++){
      XMPdimInfo d_info = subscripts.elementAt(i);
      if(d_info.isStar()){
	args = Xcons.List(descId.Ref(),
			  Xcons.IntConstant(i),Xcons.IntConstant(0),
			  Xcons.IntConstant(0),Xcons.IntConstant(0),
			  Xcons.IntConstant(0));
      } else if(!d_info.hasLower()){
	args = Xcons.List(descId.Ref(),
			  Xcons.IntConstant(i),Xcons.IntConstant(1),
			  d_info.getUpper(),Xcons.IntConstant(0),
			  Xcons.IntConstant(0));
      } else {
	args = Xcons.List(descId.Ref(),
			  Xcons.IntConstant(i),Xcons.IntConstant(2),
			  d_info.getLower(),d_info.getUpper(),
			  d_info.getStride());
      }
      bb.add(f.callSubroutine(args));
    }

    f = env.declExternIdent(XMP.ref_init_f,Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(descId.Ref())));
    
    return b;
  }

  /*
   * ref_tmpl_alloc_f(ref_id,temp_id,#n_dim)
   * ref_set_loop_info(ref_id,#dim_i,#loop_idx,off)
   * ref_init(ref_id)
   */
  public Block buildLoopConstructor(XMPenv env){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f = env.declExternIdent(XMP.ref_templ_alloc_f,Xtype.FsubroutineType);
    Xobject args = Xcons.List(descId.Ref(),refObject.getDescId().Ref(),
			      Xcons.IntConstant(subscripts.size()));
    bb.add(f.callSubroutine(args));

    f = env.declExternIdent(XMP.ref_set_loop_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < loop_dims.size(); i++){
      int idx = loop_dims.elementAt(i).getLoopOnIndex();
      Xobject off = subscripts.elementAt(idx).getOnRefOffset();
      if(off == null) off = Xcons.IntConstant(0);
      args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			Xcons.IntConstant(idx), off);
      bb.add(f.callSubroutine(args));
    }
    
    f = env.declExternIdent(XMP.ref_init_f,Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(descId.Ref())));
    
    return b;
  }

  /* (not used?)
   * loop_test#n(desc_id,#idx_val_1,#idx_val_2, ...)
   */
  public Xobject buildLoopTestFuncCall(XMPenv env, XMPinfo info){

    Ident f = env.declExternIdent(XMP.loop_test_f+info.getLoopDim(),
				  Xtype.FlogicalFunctionType);
    Xobject args = Xcons.List(descId);
    for(int i = 0; i < info.getLoopDim(); i++) 
      args.add(info.getLoopVar(i));
    return f.Call(args);
  }

  /*
   * loop_test_skip(desc_id,#dim_i,indx_val)
   */
  public Xobject buildLoopTestSkipFuncCall(XMPenv env, XMPinfo info, int k){
    Ident f = env.declExternIdent(XMP.loop_test_skip_f,
				  Xtype.FlogicalFunctionType);
    Xobject args = Xcons.List(descId,Xcons.IntConstant(k),
			      info.getLoopVar(k));
    return f.Call(args);
  }
}