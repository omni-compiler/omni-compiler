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
	XMP.errorAt(pb,"cannot find objects '" + refName + "'");
	return;
      }
      Xobject subscriptList = decl.getArg(1);
      if (subscriptList != null){
	subscripts = XMPdimInfo.parseSubscripts(subscriptList);

	if (!subscriptList.isEmptyList() && subscripts.size() != refObject.getDim()){
	  XMP.errorAt(pb, "wrong number of subscripts");
	  return;
	}

      }
      else {
	subscripts = XMPdimInfo.parseSubscripts(Xcons.List());
      }
    }

    // allocate DescId
    descId = env.declObjectId(XMP.genSym("XMP_REF_"+refName), pb);
  }

  public void setLoopDimInfo(Vector<XMPdimInfo> dims) { loop_dims = dims;}

  // make contructor
  /*
   * ref_tmpl_alloc_f(ref_id,temp_id,#n_dim)
   *    or ref_node_alloc_f(ref_id, node_id,#n_dim)
   * ref_set_dim_info(ref_id,#dim_i,ref_kind,lb,ub,step)
   * ref_init(ref_id)
   */
  /* ref_kind */
  private final static int REF_ALL   = 0;
  private final static int REF_INDEX = 1;
  private final static int REF_RANGE = 2;
  private final static int REF_RANGE_NOLB = 3;
  private final static int REF_RANGE_NOUB = 4;
  private final static int REF_RANGE_NOLBUB = 5;

  private final static Xobject DUMMY = Xcons.IntConstant(-1);

  public Block buildConstructor(XMPenv env){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f;

    switch(refObject.getKind()){
    case XMPobject.NODES:
      f = env.declInternIdent(XMP.ref_nodes_alloc_f,Xtype.FsubroutineType);
      break;
    case XMPobject.TEMPLATE:
      f = env.declInternIdent(XMP.ref_templ_alloc_f,Xtype.FsubroutineType);
      break;
    default:
      XMP.fatal("bad object for ref");
      return null;
    }
    Xobject args = Xcons.List(descId.Ref(),refObject.getDescId().Ref(),
			      Xcons.IntConstant(subscripts.size()));
    bb.add(f.callSubroutine(args));
    f = env.declInternIdent(XMP.ref_set_dim_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < subscripts.size(); i++){
      XMPdimInfo d_info = subscripts.elementAt(i);
      Xobject stride = d_info.getStride();
      if(d_info.isStar()){
	args = Xcons.List(descId.Ref(),
			  Xcons.IntConstant(i),Xcons.IntConstant(REF_ALL),
			  Xcons.IntConstant(0),Xcons.IntConstant(0),
			  Xcons.IntConstant(0));
      }
      else if (d_info.isScalar()){ // scalar
	  args = Xcons.List(descId.Ref(),
			    Xcons.IntConstant(i),Xcons.IntConstant(REF_INDEX),
			    d_info.getIndex(),Xcons.IntConstant(0),
			    Xcons.IntConstant(0));
      }
      else { // triplet
	Xobject lower, upper;
	boolean noLB_flag = false, noUB_flag = false;

	if (d_info.hasLower()){
	  lower = d_info.getLower();
	}
	else {

	  Xobject decl_lower;

	  if (refObject.getKind() == XMPobject.NODES){
	    decl_lower = Xcons.IntConstant(1);
	  }
	  else { // XMPobject.TEMPLATE
	    decl_lower = ((XMPtemplate)refObject).getLowerAt(i);
	    if (decl_lower == null){ // if not fixed
	      decl_lower = DUMMY;
	      noLB_flag = true;
	    }
	  }

	  lower = decl_lower;

	}

	if (d_info.hasUpper()){
	  upper = d_info.getUpper();
	}
	else {

	  Xobject decl_upper;

	  if (refObject.getKind() == XMPobject.NODES){
	    decl_upper = ((XMPnodes)refObject).getInfoAt(i).getUpper();
	    if (decl_upper == null){
	      decl_upper = DUMMY;
	      noUB_flag = true;
	    }
	  }
	  else { // XMPobject.TEMPLATE
	    decl_upper = ((XMPtemplate)refObject).getUpperAt(i);
	    if (decl_upper == null){ // if not fixed
	      decl_upper = DUMMY;
	      noUB_flag = true;
	    }

	  }

	  upper = decl_upper;

	}

	stride = d_info.getStride();

	int refType;
	if (!noLB_flag && !noUB_flag) refType = REF_RANGE;
	else if (noLB_flag && !noUB_flag) refType = REF_RANGE_NOLB;
	else if (!noLB_flag && noUB_flag) refType = REF_RANGE_NOUB;
	else refType = REF_RANGE_NOLBUB;

	args = Xcons.List(descId.Ref(),
			  Xcons.IntConstant(i),Xcons.IntConstant(refType),
			  lower, upper, stride);
      }
      // } else if(!d_info.hasLower()){
      // 	args = Xcons.List(descId.Ref(),
      // 			  Xcons.IntConstant(i),Xcons.IntConstant(REF_INDEX),
      // 			  d_info.getUpper(),Xcons.IntConstant(0),
      // 			  Xcons.IntConstant(0));
      // } else {
      // 	args = Xcons.List(descId.Ref(),
      // 			  Xcons.IntConstant(i),Xcons.IntConstant(REF_RANGE),
      // 			  d_info.getLower(),d_info.getUpper(),
      // 			  d_info.getStride());
      // }
      bb.add(f.callSubroutine(args));
    }

    f = env.declInternIdent(XMP.ref_init_f,Xtype.FsubroutineType);
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
    Ident f = env.declInternIdent(XMP.ref_templ_alloc_f,Xtype.FsubroutineType);
    Xobject args = Xcons.List(descId.Ref(),refObject.getDescId().Ref(),
			      Xcons.IntConstant(subscripts.size()));
    bb.add(f.callSubroutine(args));

    f = env.declInternIdent(XMP.ref_set_loop_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < loop_dims.size(); i++){
      int idx = loop_dims.elementAt(i).getLoopOnIndex();
      Xobject off = subscripts.elementAt(idx).getOnRefOffset();
      if(off == null) off = Xcons.IntConstant(0);
      args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			Xcons.IntConstant(idx), off);
      bb.add(f.callSubroutine(args));
    }
    
    f = env.declInternIdent(XMP.ref_init_f,Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(descId.Ref())));
    
    return b;
  }

  /* (not used?)
   * loop_test#n(desc_id,#idx_val_1,#idx_val_2, ...)
   */
  public Xobject buildLoopTestFuncCall(XMPenv env, XMPinfo info){

    Ident f = env.declInternIdent(XMP.loop_test_f+info.getLoopDim(),
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
    Ident f = env.declInternIdent(XMP.loop_test_skip_f,
				  Xtype.FlogicalFunctionType);
    Xobject args = Xcons.List(descId,Xcons.IntConstant(k),
			      info.getLoopVar(k));
    return f.Call(args);
  }

  public XMPobjectsRef convertLoopToReduction(){
    XMPobjectsRef new_obj = new XMPobjectsRef();
    new_obj.descId = descId;
    new_obj.refName = refName;
    new_obj.refObject = refObject;
    new_obj.subscripts = new Vector<XMPdimInfo>();

    for (int i = 0; i < subscripts.size(); i++){
      XMPdimInfo sub_i = subscripts.elementAt(i);
      XMPdimInfo newDim = new XMPdimInfo();
      if (sub_i.isTriplet()){
	newDim.setStar();
      }
      else {
	newDim.setLower(sub_i.getLower());
	newDim.setUpper(sub_i.getUpper());
	newDim.setStride(sub_i.getStride());
	if (sub_i.isStar()) newDim.setStar();
      }
      new_obj.subscripts.addElement(newDim);
    }

    for (int i = 0; i < loop_dims.size(); i++){
      XMPdimInfo loop_i = loop_dims.elementAt(i);
      int k;
      if ((k = loop_i.getLoopOnIndex()) != -1){
	new_obj.subscripts.elementAt(k).setLower(loop_i.getLower());
	new_obj.subscripts.elementAt(k).setUpper(loop_i.getUpper());
	new_obj.subscripts.elementAt(k).setStride(loop_i.getStride());
      }
    }

    return new_obj;
  }
}
