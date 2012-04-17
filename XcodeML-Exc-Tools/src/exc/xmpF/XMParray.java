/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.block.*;
import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

/*
 * XMP distribured array object
 */
public class XMParray {
  private String		name;
  private Xtype			type;
  private Xtype elementType;
  private Xtype localType;

  private Vector<XMPdimInfo> dims;

  private Ident			arrayId; // global/original ident
  private Ident			descId;  // descriptor
  private Ident 		localId; // local ident

  private XMPtemplate		template;

  // null constructor
  public XMParray() { }

  public String toString(){
    String s = "{Array("+name+"):";
    s += dims;
    return s+"}";
  }

  public String getName() {
    return name;
  }

  public Xtype getType() {
    return type;
  }

  public int getDim(){
    return dims.size();
  }

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, 
				       int index) {
    dims.elementAt(index).align_subscript_index = alignSubscriptIndex;
  }

  public Integer getAlignSubscriptIndexAt(int index) {
    return dims.elementAt(index).align_subscript_index;
  }

  public void setAlignSubscriptExprAt(Xobject alignSubscriptExpr, 
				      int index) {
    dims.elementAt(index).align_subscript_expr = alignSubscriptExpr;
  }

  public Xobject getAlignSubscriptExprAt(int index) {
    return dims.elementAt(index).align_subscript_expr;
  }

  public void setShadow(int left, int right, int index) {
    dims.elementAt(index).shadow_left = left;
    dims.elementAt(index).shadow_right = right;
  }

  public void setFullShadow(int index) {
    dims.elementAt(index).is_full_shadow = true;
  }

  public boolean isFullShadow(int index){
    return dims.elementAt(index).is_full_shadow;
  }

  public boolean hasShadow(int index){
    return (dims.elementAt(index).is_full_shadow ||
	    dims.elementAt(index).shadow_left != 0 ||
	    dims.elementAt(index).shadow_right != 0);
  }

  public boolean hasShadow(){
    for(int i = 0; i < dims.size(); i++){
      if(hasShadow(i)) return true;
    }
    return false;
  }

  public int getShadowLeft(int index) {
    return dims.elementAt(index).shadow_left;
  }

  public int getShadowRight(int index) {
    return dims.elementAt(index).shadow_right;
  }

  public Ident getArrayId() {
    return arrayId;
  }

  public Ident getDescId() {
    return descId;
  }

  public Ident getLocalId(){
    return localId;
  }


  public XMPtemplate getAlignTemplate() {
    return template;
  }

  /* 
   * Method to translate align directive 
   */
  public static void analyzeAlign(Xobject a, Xobject arrayArgs,
				  Xobject templ, Xobject tempArgs,
				  XMPenv env, PragmaBlock pb){
    XMParray arrayObject = new XMParray();
    arrayObject.parseAlign(a,arrayArgs,templ,tempArgs,env,pb);
    env.putXMParray(arrayObject,pb);
  }

  void parseAlign(Xobject a, Xobject alignSourceList,
		  Xobject templ, Xobject alignScriptList,
		  XMPenv env, PragmaBlock pb){
    Xobject t,tt;

    // get array information
    name = a.getString();
    XMParray array = env.getXMParray(name, pb);
    if (array != null) {
      XMP.error("array '" + name + "' is already aligned");
      return;
    }

    arrayId = env.findVarIdent(name, pb);
    if (arrayId == null) {
      XMP.error("array '" + name + "' is not declared");
      return;
    }
    
//     if(arrayId.getStorageClass() != StorageClass.PARAM &&
//        arrayId.getStorageClass() != StorageClass.EXTDEF &&
//        arrayId.getStorageClass() != StorageClass.EXTERN){
//       Sytem.out.println("sclass="+arrayId.getStorageClass());
//           throw new XMPexception("array '" + arrayName + 
// 				 "' must be external/global array or  a parameter of a function");
//     }
    if(XMP.debugFlag) System.out.println("arrayId="+arrayId);
    
    type  = arrayId.Type();
    if (type.getKind() != Xtype.F_ARRAY) {
      XMP.error(name + " is not an array");
      return;
    }

    // get template information
    String templateName = templ.getString();
    template = env.getXMPtemplate(templateName, pb);

    if (template == null) {
      XMP.error("template '" + templateName + "' is not declared");
    }

    if (!template.isFixed()) {
      XMP.error("template '" + templateName + "' is not fixed");
    }

    if (!(template.isDistributed())) {
      XMP.error("template '" + templateName + "' is not distributed");
    }

    if(XMP.hasError()) return;

    int templateDim = template.getDim();

    int arrayDim = type.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      XMP.error("array dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    // declare array address pointer, array descriptor
    descId = env.declObjectId(XMP.DESC_PREFIX_ + name, pb);
    elementType = type.getRef();
    
    Xobject sizeExprs[] = new Xobject[arrayDim];
    for(int i = 0; i < arrayDim; i++)
      sizeExprs[i] = Xcons.FindexRangeOfAssumedShape();
    //localType = new FarrayType(null,elementType,0,sizeExprs);
    localType = Xtype.Farray(elementType,sizeExprs);
    localType.setTypeQualFlags(Xtype.TQ_FALLOCATABLE);
    localId = env.declObjectId(XMP.PREFIX_+name,localType,pb); //the same type?

    Vector<XMPdimInfo> src_dims = XMPdimInfo.parseSubscripts(alignSourceList);
    Vector<XMPdimInfo> tmpl_dims = XMPdimInfo.parseSubscripts(alignScriptList);

    // check src_dims
    for(XMPdimInfo i: src_dims){
      if(i.isStar()) continue;
      if(i.isTriplet())
	XMP.error("bad syntax in align source script");
      t = i.getIndex();
      if(t.isVariable()){
	for(XMPdimInfo j: src_dims){  // cross check!
	  if(j.isStar()) continue;
	  if(t != j.getIndex() && t.equals(j.getIndex())){
	    XMP.error("same variable is found for '"+t.getName()+"'");
	    break;
	  }
	}
	if(XMP.hasError()) break;
      } else 
	XMP.error("align source script must be variable");
    }

    // check tmpl_dims
    for(XMPdimInfo i: tmpl_dims){
      if(i.isStar()) continue;
      if(i.isTriplet())
	XMP.error("bad syntax in align script");
      t = i.getIndex();
      if(!t.isVariable()){
	switch(t.Opcode()){
	case PLUS_EXPR:
	case MINUS_EXPR:
	  if(!t.left().isVariable())
	    XMP.error("left hand-side in align-subscript must be a variable");
	  // check right-hand side?
	  break;
	default:
	  XMP.error("bad expression in align-subsript");
	}
      }
    }

    if(src_dims.size() != arrayDim){
      XMP.error("source dimension is different from array dimension");
    }
      
    if(XMP.hasError()) return;

    // allocate dims
    dims = new Vector<XMPdimInfo>();
    for(Xobject x: type.getFarraySizeExpr())
      dims.add(XMPdimInfo.createFromRange(x));

    for(int i = 0; i < src_dims.size(); i++){
      XMPdimInfo d_info = src_dims.elementAt(i);
      if(d_info.isStar()) {
	dims.elementAt(i).setAlignSubscript(-1,null);
	continue;
      }
      t = d_info.getIndex(); // must be variable
      // find the associated variable in align-script
      int idx = -1;
      Xobject idxOffset = null;
      for(int j = 0; j < tmpl_dims.size(); j++){
	tt = tmpl_dims.elementAt(j).getIndex();
	if(tt.isVariable()){
	  if(tt.equals(t)){
	    idx = j;
	    break;
	  } 
	}  else if(tt.left().equals(t)){
	  idx = j;
	  idxOffset = tt.right();
	  if(tt.Opcode() == Xcode.MINUS_EXPR)
	    idxOffset = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,idxOffset);
	  break;
	}
      }

      if(idx < 0)
	XMP.error("the associated align-subscript not found:"+t.getName());
      else
	dims.elementAt(i).setAlignSubscript(idx,idxOffset);
    }
  }

  public static void analyzeShadow(Xobject a, Xobject shadow_w_list,
				   XMPenv env, PragmaBlock pb){
    if(!a.isVariable()){
      XMP.error("shadow cannot applied to non-array");
      return;
    }
    String name = a.getString();
    XMParray array = env.getXMParray(name, pb);
    if (array == null) {
      XMP.error("array '" + name + "'for shadow  is not declared");
      return;
    }
    Vector<XMPdimInfo> dims = XMPdimInfo.parseSubscripts(shadow_w_list);
    if(dims.size() != array.getDim()){
      XMP.error("shadow dimension size is different from array dimension");
      return;
    }
    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo d_info = dims.elementAt(i);
      int right = 0;
      int left = 0;
      if(d_info.isStar())
	array.setFullShadow(i);
      else {
	if(d_info.hasStride()){
	  XMP.error("bad syntax in shadow");
	  continue;
	}
	if(d_info.getLower() != null){
	  if(d_info.getLower().isIntConstant())
	    left = d_info.getLower().getInt();
	  else
	    XMP.error("shadow width(right) is not integer constant");
	  if(d_info.getUpper().isIntConstant())
	    right = d_info.getUpper().getInt();
	  else
	    XMP.error("shadow width(left) is not integer constant");
	} else {
	  if(d_info.getIndex().isIntConstant())
	    left = right = d_info.getIndex().getInt();
	  else 
	    XMP.error("shadow width is not integer constant");
	}
	array.setShadow(left,right,i);
      }
    }
  }

  /* !$xmp align A(i) with t(i+off)
   *
   *  ! _xmpf_array_alloc(a_desc,#dim,type,t_desc)
   *  ! _xmpf_array_range__(a_desc,i_dim,lower_b,upper_b,t_idx,off)
   *  ! _xmpf_array_init__(a_desc)
   *
   *  ! _xmpf_array_get_local_size(a_desc,i_dim,size)
   *  allocate ( A_local(0:a_1_size-1, 0:...) )
   *  ! _xmpf_array_set_local_array(a_desc,a_local)
   */

  public Block buildConstructor(XobjectDef def){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f;
    Xobject args;
    
    f = def.declExternIdent(XMP.array_alloc_f,Xtype.FsubroutineType);
    args = Xcons.List(descId.Ref(),Xcons.IntConstant(dims.size()),
		     XMP.typeIntConstant(elementType),
		     template.getDescId().Ref());
    bb.add(f.callSubroutine(args));

    f = def.declExternIdent(XMP.array_align_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo info = dims.elementAt(i);
      if(info.isAlignAny()){
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  info.getLower(),info.getUpper(),
			  Xcons.IntConstant(-1),
			  Xcons.IntConstant(0));
      } else {
	Xobject off = info.getAlignSubscriptExpr();
	if(off == null) off = Xcons.IntConstant(0);
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  info.getLower(),info.getUpper(),
			  Xcons.IntConstant(info.getAlignSubscriptIndex()),
			  off);
      }
      bb.add(f.callSubroutine(args));
    }

    if(hasShadow()){
      f = def.declExternIdent(XMP.array_init_shadow_f,Xtype.FsubroutineType);
      for(int i = 0; i < dims.size(); i++){
	if(hasShadow(i)){
	  int left = getShadowLeft(i);
	  int right = getShadowRight(i);
	  if(isFullShadow(i)) left = right = -1;
	  args = Xcons.List(descId.Ref(),
			    Xcons.IntConstant(i),
			    Xcons.IntConstant(left),
			    Xcons.IntConstant(right));
	  bb.add(f.callSubroutine(args));
	}
      }
    }

    f = def.declExternIdent(XMP.array_init_f,Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(descId.Ref())));

    // allocate size variable
    XobjList alloc_args = Xcons.List();
    for(int i = 0; i < dims.size(); i++){
      Ident ub = Ident.Local(XMP.genSym("XMP_ub_"),Xtype.FintType);
      Ident lb = Ident.Local(XMP.genSym("XMP_lb_"),Xtype.FintType);
      f = def.declExternIdent(XMP.array_get_local_size_f,
			      Xtype.FsubroutineType);
      bb.add(f.callSubroutine(Xcons.List(descId.Ref(),
					 Xcons.IntConstant(i),
					 lb.Ref(),ub.Ref())));
      alloc_args.add(Xcons.FindexRange(lb.Ref(),ub.Ref()));
    }

    // allocatable
    bb.add(Xcons.FallocateByList(localId.Ref(),alloc_args));
    
    // set
    f = def.declExternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(descId.Ref(),localId.Ref())));

    return b;
  }
}
