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

  
  private boolean		hasShadow;
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

  public Ident getArrayId() {
    return arrayId;
  }

  public Ident getDescId() {
    return descId;
  }

  public Ident getLocalId(){
    return localId;
  }

  public void setHasShadow() {
    hasShadow = true;
  }

  public boolean hasShadow() {
    return hasShadow;
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
    System.out.println("align="+arrayObject);
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
    System.out.println("arrayId="+arrayId);
    
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

    // declare array address pointer, array descriptor
    descId = env.declObjectId(XMP.DESC_PREFIX_ + name, pb);
    elementType = type.getRef();
    localType = Xtype.Farray(elementType,Xcons.FindexRangeOfAssumedShape());
    localType.setTypeQualFlags(Xtype.TQ_FALLOCATABLE);
    localId = env.declObjectId(XMP.PREFIX_+name,localType,pb); // the same type?

    int arrayDim = type.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      XMP.error("array dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    Vector<XMPdimInfo> src_dims = XMPdimInfo.parseSubscripts(alignSourceList);
    Vector<XMPdimInfo> tmpl_dims = XMPdimInfo.parseSubscripts(alignScriptList);

    // check src_dims
    for(XMPdimInfo i: src_dims){
      if(i.isStar()) continue;
      if(i.isTriplet())
	XMP.error("bad syntax in align source script");
      t = i.getIndex();
      if(t.isVariable()){
	for(XMPdimInfo j: src_dims)  // cross check!
	  if(t != j.getIndex() && t.equals(j.getIndex())){
	    XMP.error("same variable is found for '"+t.getName()+"'");
	    break;
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
      t = src_dims.elementAt(i).getIndex(); // must be variable
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

  /* !$xmp align A(i) with t(i+off)
   *
   *  ! _xmpf_array_alloc(a_desc,#dim,type,t_desc)
   *  ! _xmpf_array_range__(a_desc,i_dim,lower_b,upper_b,t_idx,off)
   *  ! _xmpf_array_init__(a_desc)
   *
   *  ! _xmpf_array_get_local_size(a_desc,size)
   *  allocate ( A_local(0:a_1_size-1) )
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
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(args)));

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
      bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(args)));
    }

    f = def.declExternIdent(XMP.array_init_f,Xtype.FsubroutineType);
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(Xcons.List(descId.Ref()))));

    // allocate size variable
    Ident size_v = Ident.Local(XMP.genSym("XMP_size_"),Xtype.FintType);
    f = def.declExternIdent(XMP.array_get_local_size_f,Xtype.FsubroutineType);
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(Xcons.List(descId.Ref(),size_v.Ref()))));
    
    // allocatable
    bb.add(Xcons.Fallocate(localId.Ref(),
			   Xcons.FindexRange(Xcons.IntConstant(0),size_v.Ref())));
    
    // set
    f = def.declExternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType);
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(Xcons.List(descId.Ref(),localId.Ref()))));

    return b;
  }
}
