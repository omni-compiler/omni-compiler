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
  private String name;  // original name
  private Xtype	type;   // original type
  private Xtype elementType;
  private Xtype localType;

  private Vector<XMPdimInfo> dims;

  private Ident	arrayId; // global/original ident
  private Ident	descId;  // descriptor

  // private Ident localId; // local ident
  
  private XMPtemplate template;

  StorageClass sclass;

  // null constructor
  public XMParray() { }

  public String toString(){
    String s = "{Array("+name+", id="+arrayId+"):";
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

  public boolean isDistributed(int index){
    int idx = dims.elementAt(index).getAlignSubscriptIndex();
    if(idx < 0) return false; // not distributed
    return template.getDistMannerAt(idx) != XMPtemplate.DUPLICATION;
  }

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, 
				       int index) {
    dims.elementAt(index).align_subscript_index = alignSubscriptIndex;
  }

  public int getAlignSubscriptIndexAt(int index) {
    return dims.elementAt(index).align_subscript_index;
  }

  public void setAlignSubscriptOffsetAt(Xobject alignSubscriptOffset, 
				      int index) {
    dims.elementAt(index).align_subscript_offset = alignSubscriptOffset;
  }

  public Xobject getAlignSubscriptOffsetAt(int index) {
    return dims.elementAt(index).align_subscript_offset;
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
    env.declXMParray(arrayObject,pb);
  }

  void parseAlign(Xobject a, Xobject alignSourceList,
		  Xobject templ, Xobject alignScriptList,
		  XMPenv env, PragmaBlock pb){
    Xobject t,tt;

    // get array information
    name = a.getString();
    XMParray array = env.findXMParray(name, pb);
    if (array != null) {
      XMP.error("array '" + name + "' is already aligned");
      return;
    }

    arrayId = pb.findVarIdent(name);  // array ident
    if (arrayId == null) {
      XMP.error("array '" + name + "' is not declared");
      return;
    }

    if(XMP.debugFlag) System.out.println("arrayId="+arrayId);

    type  = arrayId.Type();
    if (type.getKind() != Xtype.F_ARRAY) {
      XMP.error(name + " is not an array");
      return;
    }

    sclass = arrayId.getStorageClass();
    if(XMP.debugFlag) System.out.println("sclass="+sclass);
    switch(sclass){
    case PARAM:
    case EXTDEF:
    case EXTERN:
    case FLOCAL:
    case FPARAM:
    case FSAVE:
      break;
    default:
      XMP.error("bad storage class of XMP array");
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
    
    Xobject sizeExprs[] = new Xobject[1];
    switch(sclass){
    case FPARAM:
      sizeExprs[0] = Xcons.FindexRange(Xcons.IntConstant(0),
				       Xcons.IntConstant(1));
      localType = Xtype.Farray(elementType,sizeExprs);
      break;
    case FLOCAL:
    case FSAVE:
      sizeExprs[0] = Xcons.FindexRangeOfAssumedShape();
      localType = Xtype.Farray(elementType,sizeExprs);
      localType.setTypeQualFlags(type.getTypeQualFlags());
      localType.setIsFallocatable(true);
      break;
    default:
      XMP.fatal("XMP_array: unknown sclass");
    }
    // rewrite name
    String xmp_a_name = XMP.PREFIX_+name;
    arrayId.setName(xmp_a_name);
    arrayId.setType(localType);
    
    Xobject a_value = arrayId.getValue();
    if(a_value.Opcode() == Xcode.VAR && a_value.getName().equals(name))
      a_value.setName(xmp_a_name);

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
	if(tt == null) continue;
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

    // Finally, allocate size and offset var
    for(XMPdimInfo info: dims){
    	// allocate variables for size and offset
      info.setArrayInfoVar(env.declIdent(XMP.genSym("XMP_a_size_"),
					 Xtype.FintType),
			   env.declIdent(XMP.genSym("XMP_a_off_"),
					 Xtype.FintType));
    }
  }

  public static void analyzeShadow(Xobject a, Xobject shadow_w_list,
				   XMPenv env, PragmaBlock pb){
    if(!a.isVariable()){
      XMP.error("shadow cannot applied to non-array");
      return;
    }
    String name = a.getString();
    XMParray array = env.findXMParray(name, pb);
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
	if(d_info.hasLower()){
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

  public void buildConstructor(BlockList body, XMPenv env){
    Ident f;
    Xobject args;
    
    f = env.declExternIdent(XMP.array_alloc_f,Xtype.FsubroutineType);
    args = Xcons.List(descId.Ref(),Xcons.IntConstant(dims.size()),
		     XMP.typeIntConstant(elementType),
		     template.getDescId().Ref());
    body.add(f.callSubroutine(args));

    f = env.declExternIdent(XMP.array_align_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo info = dims.elementAt(i);
      if(info.isAlignAny()){
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  info.getLower(),info.getUpper(),
			  Xcons.IntConstant(-1),
			  Xcons.IntConstant(0));
      } else {
	Xobject off = info.getAlignSubscriptOffset();
	if(off == null) off = Xcons.IntConstant(0);
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  info.getLower(),info.getUpper(),
			  Xcons.IntConstant(info.getAlignSubscriptIndex()),
			  off);
      }
      body.add(f.callSubroutine(args));
    }

    if(hasShadow()){
      f = env.declExternIdent(XMP.array_init_shadow_f,Xtype.FsubroutineType);
      for(int i = 0; i < dims.size(); i++){
	if(hasShadow(i)){
	  int left = getShadowLeft(i);
	  int right = getShadowRight(i);
	  if(isFullShadow(i)) left = right = -1;
	  args = Xcons.List(descId.Ref(),
			    Xcons.IntConstant(i),
			    Xcons.IntConstant(left),
			    Xcons.IntConstant(right));
	  body.add(f.callSubroutine(args));
	}
      }
    }

    f = env.declExternIdent(XMP.array_init_f,Xtype.FsubroutineType);
    body.add(f.callSubroutine(Xcons.List(descId.Ref())));

    // allocate size variable
    Xobject alloc_size = null;
    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo info = dims.elementAt(i);
      f = env.declExternIdent(XMP.array_get_local_size_f,
			      Xtype.FsubroutineType);
      body.add(f.callSubroutine(Xcons.List(descId.Ref(),
					 Xcons.IntConstant(i),
					 info.getArraySizeVar().Ref(),
					 info.getArrayOffsetVar().Ref())));
      if(alloc_size == null)
	alloc_size = info.getArraySizeVar().Ref();
      else
	alloc_size = Xcons.binaryOp(Xcode.MUL_EXPR, 
				    alloc_size,info.getArraySizeVar().Ref());
    }

    // allocatable
    switch(sclass){
    case FLOCAL:
      body.add(Xcons.Fallocate(arrayId.Ref(),
			     Xcons.FindexRange(Xcons.IntConstant(0),
					       alloc_size)));
      break;
    case FSAVE:
      Xobject cond =  
	env.FintrinsicIdent(Xtype.FlogicalFunctionType,"allocated").
	Call(Xcons.List(arrayId.Ref()));
      body.add(Bcons.IF(Xcons.unaryOp(Xcode.LOG_NOT_EXPR,cond),
		      Xcons.Fallocate(arrayId.Ref(),
				      Xcons.FindexRange(Xcons.IntConstant(0),
							alloc_size)),null));
      break;
    }
    
    // set
    f = env.declExternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType);
    body.add(f.callSubroutine(Xcons.List(descId.Ref(),arrayId.Ref())));
  }

  public void buildSetup(BlockList body, XMPenv env){
    Ident f;
    Xobject args;
    
    // not yet
  }

  public void buildDestructor(BlockList body, XMPenv env){
    Ident f;
    Xobject args;
    
    f = env.declExternIdent(XMP.array_dealloc_f,Xtype.FsubroutineType);
    args = Xcons.List(descId.Ref());
    body.add(f.callSubroutine(args));
  }

  public Xobject convertLinearIndex(Xobject index_list){
    Xobject idx = null;
    for(int i = dims.size()-1; i >= 0; i--){
      XMPdimInfo info = dims.elementAt(i);
      Xobject x = index_list.getArg(i);
      if(x.Opcode() != Xcode.F_ARRAY_INDEX)
	 XMP.fatal("convertLinearIndex: not F_ARRAY_INDEX");
      x = x.getArg(0);
      x = Xcons.binaryOp(Xcode.MINUS_EXPR,x,info.getArrayOffsetVar().Ref());
      if(idx == null) idx = x;
      else idx = Xcons.binaryOp(Xcode.PLUS_EXPR,idx,x);
      if(i != 0) idx = Xcons.binaryOp(Xcode.MUL_EXPR,idx,
				      info.getArraySizeVar().Ref());
    }
    return Xcons.List(Xcons.FarrayIndex(idx));
  }
}
