package exc.xmpF;

import exc.block.*;
import exc.object.*;
import java.util.Vector;
import java.util.Iterator;

/*
 * XMP distribured array object
 */
public class XMParray {
  private final static String XMP_ARRAY_PROP = "XMP_ARRAY_PROP";

  private Ident	arrayId;      // global/original ident
  private Block arrayIdBlock; // Block ident found
  private String name;        // original name
  private Xtype	type;         // original type
  private Xtype elementType;
  private Ident localId; 
  private Vector<XMPdimInfo> dims;
  private Ident	descId;      // descriptor
  private XMPtemplate template;
  StorageClass sclass;
  boolean is_linearized       = false;
  private int NthAssumedShape = -1;
  boolean shadow_declared     = false;
  boolean is_saveDesc         = false;

  // null constructor
  public XMParray() { }

  public XMParray(XMParray orig, Ident newId, String newName, Ident newLocalId){
    arrayId     = newId;
    name        = newName;
    type        = orig.type;
    elementType = orig.elementType;
    localId     = newLocalId;
    dims        = orig.dims;
    descId      = orig.descId;
    template    = orig.template;
    sclass      = orig.sclass;
  }    

  public String toString(){
    String s = "{Array("+name+", id="+arrayId+"):";
    s += dims;
    s += " localId:" + localId;
    return s+"}";
  }

  public String getName() { return name; }

  public Xtype getType() { return type; }

  public int getDim(){ return dims.size(); }

  public boolean isDistributed(int index){
    int idx = dims.elementAt(index).getAlignSubscriptIndex();
    if(idx < 0) return false; // not distributed
    return template.getDistMannerAt(idx) != XMPtemplate.DUPLICATION;
  }

  public int getDistMannerAt(int index){
    int idx = dims.elementAt(index).getAlignSubscriptIndex();
    if(idx < 0) return -1; // not distributed
    return template.getDistMannerAt(idx);
  }    

  public void setAlignSubscriptIndexAt(int alignSubscriptIndex, int index) {
    dims.elementAt(index).align_subscript_index = alignSubscriptIndex;
  }

  public int getAlignSubscriptIndexAt(int index) {
    return dims.elementAt(index).align_subscript_index;
  }

  public void setAlignSubscriptOffsetAt(Xobject alignSubscriptOffset, int index) {
    dims.elementAt(index).align_subscript_offset = alignSubscriptOffset;
  }

  public Xobject getAlignSubscriptOffsetAt(int index) {
    return dims.elementAt(index).align_subscript_offset;
  }

  public Ident getBlkOffsetVarAt(int index) {
    return dims.elementAt(index).getArrayBlkOffsetVar();
  }

  public void setShadow(int left, int right, int index) {
    dims.elementAt(index).shadow_left  = left;
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
	    dims.elementAt(index).shadow_left  != 0 ||
	    dims.elementAt(index).shadow_right != 0);
  }

  public boolean hasShadow(){
    for(int i = 0; i < dims.size(); i++)
      if(hasShadow(i)) return true;
    
    return false;
  }

  public int getShadowLeft(int index) {
    return dims.elementAt(index).shadow_left;
  }

  public int getShadowRight(int index) {
    return dims.elementAt(index).shadow_right;
  }

  public Ident getSizeVarAt(int index) {
    return dims.elementAt(index).a_dim_size_var;
  }

  public Xobject getLowerAt(int index){
    return dims.elementAt(index).getLower();
  }    
    
  public Ident getArrayId() { return arrayId; }

  public Ident getDescId() { return descId; }

  public XMPtemplate getAlignTemplate() { return template; }

  public Ident getLocalId() { return localId; }

  public void setLocalId(Ident id) { localId = id; }

  public Xtype getLocalType() { return localId.Type(); }
  
  public String getLocalName() { return localId.getName(); }

  public static XMParray getArray(Xobject id){
    return (XMParray) id.getProp(XMP_ARRAY_PROP);
  }

  public static void setArray(Xobject id, XMParray array){
    id.setProp(XMP_ARRAY_PROP,array);
  }

  public void setLinearized(boolean flag){ is_linearized = flag; }

  public boolean isSaveDesc() { return is_saveDesc; }

  public void setSaveDesc(boolean flag){ is_saveDesc = flag; }

  public boolean isLinearized() { return is_linearized; }

  /* 
   * Method to translate align directive 
   */
  public static void analyzeAlign(Xobject a, Xobject arrayArgs,
				  Xobject templ, Xobject tempArgs,
				  XMPenv env, PragmaBlock pb){
    boolean isStructure  = (a.Opcode() == Xcode.LIST);
    if (isStructure) {
      declareForStructure(a,arrayArgs,templ,tempArgs,env,pb);
    }
    else {
      XMParray arrayObject = new XMParray();
      arrayObject.parseAlign(a,arrayArgs,templ,tempArgs,env,pb);
      env.declXMParray(arrayObject,pb);
    }
  }

  public static void declareForStructure(Xobject a, Xobject alignSourceList,
					 Xobject templ, Xobject alignScriptList,
					 XMPenv env, PragmaBlock pb) {
    String structName = a.getArg(0).getSym();
    String memberName = a.getArg(1).getSym();
    Ident structId = (pb == null)? env.findVarIdent(structName, pb) : pb.findVarIdent(structName);

    if (structId == null) {
      XMP.errorAt(pb,"structure '" + structName + "' is not declared");
      return;
    }
    
    Ident arrayId = structId.Type().getMemberList().getIdent(memberName);
    if (arrayId == null) {
      XMP.errorAt(pb,"structure's member '" + structName + "%" + memberName + "' is not declared");
      return;
    }
    
    arrayId.saveOrigId();
    arrayId.setProp(XMP.Env, env);
    Xtype type = arrayId.Type();
    
    String templateName = templ.getString();
    XMPtemplate template = env.findXMPtemplate(templateName, pb);
    arrayId.setProp(XMP.Template, template);
    
    if (template == null)
      XMP.errorAt(pb,"template '" + templateName + "' is not declared");
    
    if (!template.isDistributed() && !type.isFallocatable())
      XMP.errorAt(pb,"template '" + templateName + "' is not distributed");
    
    if(XMP.hasError()) return;
   
    int templateDim = template.getDim();
    int arrayDim = type.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      XMP.errorAt(pb,"array dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    Vector<XMPdimInfo> src_dims  = XMPdimInfo.parseSubscripts(alignSourceList);
    Vector<XMPdimInfo> tmpl_dims = XMPdimInfo.parseSubscripts(alignScriptList);

    // check src_dims
    for(XMPdimInfo i: src_dims){
      if(i.isStar()) continue;
      if(i.isTriplet()){
        XMP.errorAt(pb,"bad syntax in align source script");
        break;
      }
      Xobject t = i.getIndex();
      if(t.isVariable()){
        for(XMPdimInfo j: src_dims){  // cross check!
          if(j.isStar()) continue;
          if(t != j.getIndex() && t.equals(j.getIndex())){
            XMP.errorAt(pb,"same variable is found for '"+t.getName()+"'");
            break;
          }
        }
        if(XMP.hasError()) break;
      } else
        XMP.errorAt(pb,"align source script must be variable");
    }

    // check tmpl_dims
    for(XMPdimInfo i: tmpl_dims){
      if(i.isStar()) continue;
      if(i.isTriplet()){
        XMP.errorAt(pb,"bad syntax in align script");
        break;
      }
      Xobject t = i.getIndex();
      if(!t.isVariable()){
        switch(t.Opcode()){
        case PLUS_EXPR:
        case MINUS_EXPR:
          if(!t.left().isVariable())
            XMP.errorAt(pb,"left hand-side in align-subscript must be a variable");
          // check right-hand side?
          break;
        default:
          XMP.errorAt(pb,"bad expression in align-subsript");
        }
      }
    }

    if(src_dims.size() != arrayDim)
      XMP.errorAt(pb,"source dimension is different from array dimension");

    if(XMP.hasError()) return;

    String localName    = XMP.PREFIX_ + memberName;
    Xobject sizeExprs[] = new Xobject[arrayDim];
    for(int i=0;i<arrayDim;i++)
      sizeExprs[i] = Xcons.FindexRangeOfAssumedShape();

    ((FarrayType)arrayId.Type()).setFarraySizeExpr(sizeExprs);
    ((FarrayType)arrayId.Type()).setIsFallocatable(true);
    arrayId.setName(localName);
    arrayId.setMemberAligned(true);
    arrayId.setProp(XMP.AlignSourceList, alignSourceList);
    arrayId.setProp(XMP.AlignScriptList, alignScriptList);
  }
  
  void parseAlign(Xobject a, Xobject alignSourceList,
		  Xobject templ, Xobject alignScriptList,
		  XMPenv env, PragmaBlock pb){
    name    = a.getString();
    arrayId = env.findVarIdent(name, pb);
    if (arrayId == null) {
      XMP.errorAt(pb,"array '" + name + "' is not declared");
      return;
    }
    arrayIdBlock = env.findVarIdentBlock(name, pb);

    type  = arrayId.Type();
    if (type.getKind() != Xtype.F_ARRAY) {
      XMP.errorAt(pb,name + " is not an array");
      return;
    }

    if(getArray(arrayId) != null){
      XMP.errorAt(pb,"array '" + name + "' is already aligned");
      return;
    }

    if(XMP.debugFlag) System.out.println("arrayId="+arrayId);

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
      XMP.errorAt(pb,"bad storage class of XMP array");
    }

    // get template information
    String templateName = templ.getString();
    template = env.findXMPtemplate(templateName, pb);

    if (template == null)
      XMP.errorAt(pb,"template '" + templateName + "' is not declared");

    if (!template.isDistributed() && !type.isFallocatable())
      XMP.errorAt(pb,"template '" + templateName + "' is not distributed");

    if(XMP.hasError()) return;

    int templateDim = template.getDim();
    int arrayDim = type.getNumDimensions();
    if (arrayDim > XMP.MAX_DIM) {
      XMP.errorAt(pb,"array dimension should be less than " + (XMP.MAX_DIM + 1));
      return;
    }

    // declare array address pointer, array descriptor
    String desc_id_name = XMP.DESC_PREFIX_ + name;
    descId = env.declObjectId(desc_id_name, arrayIdBlock);
    elementType = type.getRef();

    Vector<XMPdimInfo> src_dims = XMPdimInfo.parseSubscripts(alignSourceList);
    Vector<XMPdimInfo> tmpl_dims = XMPdimInfo.parseSubscripts(alignScriptList);

    // check src_dims
    for(XMPdimInfo i: src_dims){
      if(i.isStar()) continue;
      if(i.isTriplet()){
	XMP.errorAt(pb,"bad syntax in align source script");
	break;
      }
      Xobject t = i.getIndex();
      if(t.isVariable()){
	for(XMPdimInfo j: src_dims){  // cross check!
	  if(j.isStar()) continue;
	  if(t != j.getIndex() && t.equals(j.getIndex())){
	    XMP.errorAt(pb,"same variable is found for '"+t.getName()+"'");
	    break;
	  }
	}
	if(XMP.hasError()) break;
      } else 
	XMP.errorAt(pb,"align source script must be variable");
    }

    // check tmpl_dims
    for(XMPdimInfo i: tmpl_dims){
      if(i.isStar()) continue;
      if(i.isTriplet()){
	XMP.errorAt(pb,"bad syntax in align script");
	break;
      }
      Xobject t = i.getIndex();
      if(!t.isVariable()){
	switch(t.Opcode()){
	case PLUS_EXPR:
	case MINUS_EXPR:
	  if(!t.left().isVariable())
	    XMP.errorAt(pb,"left hand-side in align-subscript must be a variable");
	  // check right-hand side?
	  break;
	default:
	  XMP.errorAt(pb,"bad expression in align-subsript");
	}
      }
    }

    if(src_dims.size() != arrayDim)
      XMP.errorAt(pb,"source dimension is different from array dimension");
      
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

      Xobject t = d_info.getIndex(); // must be variable
      // find the associated variable in align-script
      int idx = -1;
      Xobject idxOffset = null;
      for(int j = 0; j < tmpl_dims.size(); j++){
	Xobject tt = tmpl_dims.elementAt(j).getIndex();
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
	XMP.errorAt(pb,"the associated align-subscript not found:"+t.getName());
      else
	dims.elementAt(i).setAlignSubscript(idx,idxOffset);
    }

    // Finally, allocate size and offset var
    int dim_i = 0;
    for(XMPdimInfo info: dims){
      // allocate variables for size and offset, use fixed name
      info.setArrayInfoVar(env.declIdent(desc_id_name+"_size_"+dim_i,
					 Xtype.FintType,arrayIdBlock),
			   env.declIdent(desc_id_name+"_off_"+dim_i,
					 Xtype.FintType,arrayIdBlock),
			   env.declIdent(desc_id_name+"_blkoff_"+dim_i,
					 Xtype.FintType,arrayIdBlock));
      dim_i++;
    }

    Xtype localType = null;
    Xobject sizeExprs[];

    switch(sclass){
    case FPARAM: {
      if (type.isFallocatable())
	XMP.errorAt(pb, "allocatable dummy arrays cannot be distributed.");

      Xtype ftype = env.getCurrentDef().getDef().getNameObj().Type();
      if (ftype.isFunction() && !ftype.isFsubroutine() && type.isFassumedShape())
	XMP.fatal("assumed-shape distributed arrays in functions are not supported.");

      if (ftype.isFsubroutine() && type.isFassumedShape()){
	Xobject id_list = env.getCurrentDef().getBlock().getBody().getIdentList();
	int k = 0;
	for (Xobject i: (XobjList) id_list){
	  Ident id = (Ident)i;
	  if (id.Type().isFassumedShape()){
	    if (id.getSym().equals(name)){
	      NthAssumedShape = k;
	      break;
	    }
	    k++;
	  }
	}
	if (NthAssumedShape == -1) XMP.fatal("non-dummy argument cannot have a deferred shape.");
	if (NthAssumedShape >= XMP.MAX_ASSUMED_SHAPE)
	  XMP.fatal("too many assumed-shape arguments (MAX = " + XMP.MAX_ASSUMED_SHAPE + ").");
      }

      if (env.getCurrentDef().getDef().getParent() != null ||
	  (NthAssumedShape >= 0 && NthAssumedShape < XMP.MAX_ASSUMED_SHAPE)){ // to assumed-shape
	sizeExprs = new Xobject[arrayDim];
	for (int i = 0; i < arrayDim; i++){
	  if (isDistributed(i))
	    sizeExprs[i] = Xcons.FindexRangeOfAssumedShape(Xcons.IntConstant(0));
	  else
	    sizeExprs[i] = Xcons.FindexRangeOfAssumedShape(type.getFarraySizeExpr()[i].getArg(0));
	}
	localType = Xtype.Farray(elementType, sizeExprs);
	localType.setTypeQualFlags(type.getTypeQualFlags());
      }
      else { // now linearize it
	sizeExprs = new Xobject[1];
	sizeExprs[0] = Xcons.FindexRange(Xcons.IntConstant(0),
					 Xcons.IntConstant(1));
	localType = Xtype.Farray(elementType, sizeExprs);
	setLinearized(true);
      }
      break;

    }
    case FLOCAL:
    case FSAVE:
      sizeExprs = new Xobject[arrayDim];
      for(int i = 0; i < arrayDim; i++)
	sizeExprs[i] = Xcons.FindexRangeOfAssumedShape();
      localType = Xtype.Farray(elementType,sizeExprs);
      localType.setTypeQualFlags(type.getTypeQualFlags());
      if (!localType.isFpointer()) localType.setIsFallocatable(true);
      break;
    default:
      XMP.fatal("XMP_array: unknown sclass");
    }
    
    String localName = XMP.PREFIX_+name;
    localId = env.declIdent(localName,localType,false,arrayIdBlock);
    localId.setStorageClass(arrayId.getStorageClass());
    localId.setValue(Xcons.Symbol(Xcode.VAR,localType,localName));
    setArray(arrayId,this);
  }

  public void parseAlignForStructure(String structVarName, Ident structVarId, Block structVarBlock,
				     String memberName, Ident memberId, XMPenv env)
  {
    this.name         = memberId.getOrigId().getName();
    this.arrayId      = memberId.getOrigId();
    this.arrayIdBlock = structVarBlock;
    this.localId      = memberId;
    this.type         = this.arrayId.Type();
    if (this.type.getKind() != Xtype.F_ARRAY)
      XMP.errorAt(this.arrayIdBlock, memberName + " is not an array");

    this.sclass = structVarId.getStorageClass();
    switch(this.sclass){
    case PARAM:
    case EXTDEF:
    case EXTERN:
    case FLOCAL:
    case FPARAM:
    case FSAVE:
      break;
    default:
      XMP.errorAt(this.arrayIdBlock, "bad storage class of XMP array");
    }
    if(XMP.hasError()) return;

    this.template = (XMPtemplate)memberId.getProp(XMP.Template);

    // declare array address pointer, array descriptor
    String descIdName = XMP.DESC_STRUCT_PREFIX_ + structVarName + "_" + this.name;
    this.descId       = env.declObjectId(descIdName, this.arrayIdBlock);
    this.elementType  = this.type.getRef();

    XobjList alignSourceList     = (XobjList)memberId.getProp(XMP.AlignSourceList);
    XobjList alignScriptList     = (XobjList)memberId.getProp(XMP.AlignScriptList);
    Vector<XMPdimInfo> src_dims  = XMPdimInfo.parseSubscripts(alignSourceList);
    Vector<XMPdimInfo> tmpl_dims = XMPdimInfo.parseSubscripts(alignScriptList);

    // allocate dims
    this.dims = new Vector<XMPdimInfo>();
    for(Xobject x: this.type.getFarraySizeExpr())
      this.dims.add(XMPdimInfo.createFromRange(x));

    for(int i = 0; i < src_dims.size(); i++){
      XMPdimInfo d_info = src_dims.elementAt(i);
      if(d_info.isStar()) {
        this.dims.elementAt(i).setAlignSubscript(-1,null);
        continue;
      }

      Xobject t = d_info.getIndex(); // must be variable
      // find the associated variable in align-script
      int idx = -1;
      Xobject idxOffset = null;
      for(int j = 0; j < tmpl_dims.size(); j++){
        Xobject tt = tmpl_dims.elementAt(j).getIndex();
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
        XMP.errorAt(this.arrayIdBlock, "the associated align-subscript not found:"+t.getName());
      else
        this.dims.elementAt(i).setAlignSubscript(idx,idxOffset);
    }

    // Finally, allocate size and offset var
    int dim_i = 0;
    for(XMPdimInfo info: this.dims){
      // allocate variables for size and offset, use fixed name
      info.setArrayInfoVar(env.declIdent(descIdName+"_size_"+dim_i,
                                         Xtype.FintType,arrayIdBlock),
                           env.declIdent(descIdName+"_off_"+dim_i,
                                         Xtype.FintType,arrayIdBlock),
                           env.declIdent(descIdName+"_blkoff_"+dim_i,
                                         Xtype.FintType,arrayIdBlock));
      dim_i++;
    }

    this.localId.setStorageClass(arrayId.getStorageClass());
    setArray(arrayId, this);
  }

  public static void analyzeShadow(Xobject a, Xobject shadow_w_list,
				   XMPenv env, PragmaBlock pb){
    boolean isStructure  = (a.Opcode() == Xcode.LIST);
    if (isStructure) {
      String structName = a.getArg(0).getSym();
      String memberName = a.getArg(1).getSym();
      Ident structId = (pb == null)? env.findVarIdent(structName, pb) : pb.findVarIdent(structName);
      Ident arrayId = structId.Type().getMemberList().getIdent(XMP.PREFIX_ + memberName);
      arrayId.setProp(XMP.Shadow_w_list, shadow_w_list);
      return;
    }

    if(!a.isVariable())
      XMP.errorAt(pb,"shadow cannot applied to non-array");

    String name = a.getString();
    Ident id = env.findVarIdent(name, pb);
    if(id == null){
      XMP.errorAt(pb,"variable '" + name + "'for shadow  is not declared");
      return;
    }
    XMParray array = XMParray.getArray(id);
    if (array == null) {
      XMP.errorAt(pb,"array '" + name + "'for shadow  is not declared");
      return;
    }
    Vector<XMPdimInfo> dims = XMPdimInfo.parseSubscripts(shadow_w_list);
    if(dims.size() != array.getDim()){
      XMP.errorAt(pb,"shadow dimension size is different from array dimension");
      return;
    }

    if (array.shadow_declared){
      XMP.errorAt(pb, "variable '" + name + "' already has shadow region");
    }

    array.shadow_declared = true;

    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo d_info = dims.elementAt(i);
      int right = 0;
      int left = 0;
      if(d_info.isStar())
	array.setFullShadow(i);
      else {
	if (d_info.isScalar()){ // scalar
	  if (d_info.getIndex().isIntConstant())
	    left = right = d_info.getIndex().getInt();
	  else 
	    XMP.errorAt(pb,"shadow width is not integer constant");
	}
	else if (!d_info.hasStride()){ // "lshadow : ushadow"
	  if (d_info.hasLower()){
	    if (d_info.getLower().isIntConstant())
	      left = d_info.getLower().getInt();
	    else
	      XMP.errorAt(pb,"shadow width(left) is not integer constant");
	  }
	  else {
	    XMP.errorAt(pb,"no shadow width(left) is specified.");
	  }
	  if (d_info.hasLower()){
	    if (d_info.getUpper().isIntConstant())
	      right = d_info.getUpper().getInt();
	    else
	      XMP.errorAt(pb,"shadow width(right) is not integer constant");
	  }
	  else {
	    XMP.errorAt(pb,"no shadow width(right) is specified.");
	  }
	}
	else {
	  XMP.errorAt(pb,"bad syntax in shadow");
	  continue;
	}

	array.setShadow(left,right,i);
      }
    }
  }

  public void analyzeShadowForStructure(Block b) {
    XMParray array          = XMParray.getArray(this.arrayId);
    Xobject shadow_w_list   = (Xobject)this.localId.getProp(XMP.Shadow_w_list);
    Vector<XMPdimInfo> dims = XMPdimInfo.parseSubscripts(shadow_w_list);
    if(dims.size() != array.getDim()){
      XMP.errorAt(b,"shadow dimension size is different from array dimension");
      return;
    }
    else if (array.shadow_declared){
      XMP.errorAt(b, "variable '" + name + "' already has shadow region");
      return;
    }
    array.shadow_declared = true;

    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo d_info = dims.elementAt(i);
      int right = 0;
      int left  = 0;
      
      if(d_info.isStar())
        array.setFullShadow(i);
      else {
	if (d_info.isScalar()){ // scalar
          if (d_info.getIndex().isIntConstant())
            left = right = d_info.getIndex().getInt();
          else
            XMP.errorAt(b,"shadow width is not integer constant");
        }
	else if (!d_info.hasStride()){ // "lshadow : ushadow"
          if (d_info.hasLower()){
            if (d_info.getLower().isIntConstant())
              left = d_info.getLower().getInt();
            else
              XMP.errorAt(b,"shadow width(left) is not integer constant");
          }
          else {
            XMP.errorAt(b,"no shadow width(left) is specified.");
          }
          if (d_info.hasLower()){
            if (d_info.getUpper().isIntConstant())
              right = d_info.getUpper().getInt();
            else
              XMP.errorAt(b,"shadow width(right) is not integer constant");
          }
          else {
            XMP.errorAt(b,"no shadow width(right) is specified.");
          }
        }
        else {
          XMP.errorAt(b,"bad syntax in shadow");
          continue;
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
  public void buildConstructor(BlockList body, XMPenv env, Block block){
    if (is_saveDesc && type.isFallocatable())
      XMP.fatal("an allocatable array cannot have the save_desc attribute.");

    BlockList b = (is_saveDesc && !env.currentDefIsModule())? Bcons.emptyBody() : body;
    Ident flagVar = null;
    if (is_saveDesc && !env.currentDefIsModule()){
      Xtype save_desc = descId.Type().copy();
      save_desc.setIsFsave(true);
      descId.setType(save_desc);

      for (int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);

	Xtype t0 = info.getArraySizeVar().Type().copy();
	t0.setIsFsave(true);
	info.getArraySizeVar().setType(t0);

	Xtype t1 = info.getArrayOffsetVar().Type().copy();
	t1.setIsFsave(true);
	info.getArrayOffsetVar().setType(t1);

	Xtype t2 = info.getArrayBlkOffsetVar().Type().copy();
	t2.setIsFsave(true);
	info.getArrayBlkOffsetVar().setType(t2);
      }

      Xtype save_logical = Xtype.FlogicalType.copy();
      save_logical.setIsFsave(true);
      BlockList bl = env.getCurrentDef().getBlock().getBody();
      flagVar = bl.declLocalIdent(XMP.SAVE_DESC_PREFIX_ + name, save_logical,
      				    StorageClass.FSAVE,
      				    Xcons.List(Xcode.F_VALUE, Xcons.FlogicalConstant(false)));
    }

    Ident f = env.declInternIdent(XMP.array_alloc_f,Xtype.FsubroutineType, block);
    Xobject type_size = null;
    if (!elementType.isBasic()){
      env.useModule("iso_c_binding");
      env.findModule("iso_c_binding"); // import
      XobjString modName = Xcons.Symbol(Xcode.IDENT, "iso_c_binding");
      Xobject use = Xcons.List(Xcode.F_USE_DECL, modName, Xcons.IntConstant(0));
      env.getCurrentDef().getDef().getFuncDecls().add(use);

      Ident g = env.FintrinsicIdent(Xtype.FintFunctionType, "c_sizeof");
      XobjList subs = Xcons.List();
      for (int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);
	Xobject lower = info.getLower();
	subs.add(lower);
      }
      type_size = g.Call(Xcons.List(Xcons.FarrayRef(arrayId.Ref(), subs)));
    }
    else {
      type_size = Xcons.IntConstant(0);
    }
    
    Xobject args = Xcons.List(descId.Ref(),Xcons.IntConstant(dims.size()),
			      XMP.typeIntConstant(elementType), type_size,
			      template.getDescId().Ref());
    b.add(f.callSubroutine(args));

    if(type.isFallocatable() || type.isFpointer()) return;
    
    f = env.declInternIdent(XMP.init_allocated_f, Xtype.FsubroutineType, block);
    args = Xcons.List(descId.Ref());
    b.add(f.callSubroutine(args));

    Ident sizeArray = null;
    if (type.isFassumedShape())
      sizeArray = env.declOrGetSizeArray(block);

    f = env.declInternIdent(XMP.array_align_info_f,Xtype.FsubroutineType, block);
    for(int i = 0; i < dims.size(); i++){
      XMPdimInfo info = dims.elementAt(i);
      Xobject lower = info.getLower();
      Xobject upper = null;

      if (info.getUpper() == null && NthAssumedShape >= 0 && NthAssumedShape < XMP.MAX_ASSUMED_SHAPE){
	upper = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(NthAssumedShape), Xcons.IntConstant(i));
	upper = Xcons.binaryOp(Xcode.PLUS_EXPR, upper, lower);
	upper = Xcons.binaryOp(Xcode.MINUS_EXPR, upper, Xcons.IntConstant(1));
      }
      else {
	upper = info.getUpper();
      }

      if(info.isAlignAny()){
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  lower, upper,
			  Xcons.IntConstant(-1),
			  Xcons.IntConstant(0));
      } else {
	Xobject off = info.getAlignSubscriptOffset();
	if(off == null) off = Xcons.IntConstant(0);
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  lower, upper,
			  Xcons.IntConstant(info.getAlignSubscriptIndex()),
			  off);
      }
      b.add(f.callSubroutine(args));
    }

    if(hasShadow()){
      f = env.declInternIdent(XMP.array_init_shadow_f,Xtype.FsubroutineType, block);
      for(int i = 0; i < dims.size(); i++){
	if(hasShadow(i)){
	  int left = getShadowLeft(i);
	  int right = getShadowRight(i);
	  if(isFullShadow(i)) left = right = -1;
	  args = Xcons.List(descId.Ref(),
			    Xcons.IntConstant(i),
			    Xcons.IntConstant(left),
			    Xcons.IntConstant(right));
	  b.add(f.callSubroutine(args));
	}
      }
    }

    f = env.declInternIdent(XMP.array_init_f,Xtype.FsubroutineType, block);
    b.add(f.callSubroutine(Xcons.List(descId.Ref())));

    Xobject allocate_statement = null;

    if(isLinearized()){
      // allocate size variable
      Xobject alloc_size = null;
      for(int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);
	f = env.declInternIdent(XMP.array_get_local_size_f,
			      Xtype.FsubroutineType, block);

	b.add(f.callSubroutine(Xcons.List(descId.Ref(),
					  Xcons.IntConstant(i),
					  info.getArraySizeVar().Ref(),
					  info.getArrayOffsetVar().Ref(),
					  info.getArrayBlkOffsetVar().Ref())));
	if(alloc_size == null)
	  alloc_size = info.getArraySizeVar().Ref();
	else
	  alloc_size = Xcons.binaryOp(Xcode.MUL_EXPR, 
				      alloc_size,info.getArraySizeVar().Ref());
      }
      
      // allocatable
      Xobject size_1 = Xcons.binaryOp(Xcode.MINUS_EXPR,alloc_size,
				      Xcons.IntConstant(1));
      allocate_statement = 
	Xcons.Fallocate(localId.Ref(),
			Xcons.FindexRange(Xcons.IntConstant(0),size_1));
    } else {
      XobjList alloc_args = Xcons.List();
      for(int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);
	f = env.declInternIdent(XMP.array_get_local_size_f,
				Xtype.FsubroutineType, block);
	b.add(f.callSubroutine(Xcons.List(descId.Ref(),
					  Xcons.IntConstant(i),
					  info.getArraySizeVar().Ref(),
					  info.getArrayOffsetVar().Ref(),
					  info.getArrayBlkOffsetVar().Ref())));
	if (isDistributed(i)){
	    // distributed
	    Xobject size_1 = Xcons.binaryOp(Xcode.MINUS_EXPR,
					    info.getArraySizeVar().Ref(),
					    Xcons.IntConstant(1));
	    alloc_args.add(Xcons.FindexRange(Xcons.IntConstant(0),size_1));
	}
	else {
	    // not distributed
	    alloc_args.add(Xcons.FindexRange(info.getLower(),
					     info.getArraySizeVar().Ref()));
	}
      }

      if (localId.isCoarray()){
	Xobject coShape = Xcons.List(Xcode.F_CO_SHAPE, localId.Type().getCodimensions());
	alloc_args.add(coShape);
      }

      // allocatable
      if (localId.isMemberAligned()){
	Ident structVarId       = (Ident)localId.getProp(XMP.StructId);
	Xobject structMemberRef = Xcons.List(Xcode.MEMBER_REF, type, Xcons.FvarRef(structVarId),
					     Xcons.Symbol(Xcode.IDENT, type, localId.getName()));
	allocate_statement = Xcons.FallocateByList(structMemberRef,alloc_args);
      }
      else
	allocate_statement = Xcons.FallocateByList(localId.Ref(),alloc_args);
    }

    if (is_saveDesc && !env.currentDefIsModule()){
      b.add(Xcons.Set(flagVar.Ref(), Xcons.FlogicalConstant(true)));
      body.add(Bcons.IF(BasicBlock.Cond(Xcons.unaryOp(Xcode.LOG_NOT_EXPR, flagVar.Ref())), b, null));
    }

    switch(sclass){
    case FLOCAL:
      body.add(allocate_statement);
      break;
    case FSAVE:
      {
	Xobject target = null;
	if (localId.isMemberAligned())
	  target = Xcons.List(Xcode.MEMBER_REF, type, Xcons.FvarRef((Ident)localId.getProp(XMP.StructId)),
			      Xcons.Symbol(Xcode.IDENT, type, localId.getName()));
	else
	  target = localId.Ref();
	
	Xobject cond = env.FintrinsicIdent(Xtype.FlogicalFunctionType,"allocated").
	  Call(Xcons.List(target));
	body.add(Bcons.IF(Xcons.unaryOp(Xcode.LOG_NOT_EXPR,cond),
			  allocate_statement,null));
	break;
      }
    }
    
    // set
    f = env.declInternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType, block);
    Xobject isCoarray = (sclass == StorageClass.FSAVE) ? Xcons.IntConstant(1) : Xcons.IntConstant(0);
    if (localId.isMemberAligned()){
      Ident structVarId       = (Ident)localId.getProp(XMP.StructId);
      Xobject structMemberRef = Xcons.List(Xcode.MEMBER_REF, type, Xcons.FvarRef(structVarId),
					   Xcons.Symbol(Xcode.IDENT, type, localId.getName()));
      body.add(f.callSubroutine(Xcons.List(descId.Ref(), structMemberRef, isCoarray)));
    }
    else
      body.add(f.callSubroutine(Xcons.List(descId.Ref(), localId.Ref(), isCoarray)));
  }

  /*
   * rewrite Allocate for aligned arrays
   */
    public void rewriteAllocate(XobjList alloc, Statement st,
				Block block, XMPenv env){

    XobjList boundList = (XobjList)alloc.getArg(1);
    Ident f;
    Xobject args;
    Xobject orig_lower[] = new Xobject[dims.size()];

    // Following codes come from XMParray.buildConstructor
    f = env.declInternIdent(XMP.init_allocated_f, Xtype.FsubroutineType, block);
    args = Xcons.List(descId.Ref());
    st.insert(f.callSubroutine(args));

    f = env.declInternIdent(XMP.array_align_info_f,Xtype.FsubroutineType, block);
    for(int i = 0; i < dims.size(); i++){
      XobjList bound = (XobjList)boundList.getArg(i);
      Xobject lower, upper;

      if (bound.Nargs() == 1){
	lower = Xcons.IntConstant(1);
	upper = bound.getArg(0);
      }
      else {
	lower = bound.getArg(0);
	upper = bound.getArg(1);
      }

      orig_lower[i] = lower;
      
      XMPdimInfo info = dims.elementAt(i);
      if(info.isAlignAny()){
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  lower, upper, // modified
			  Xcons.IntConstant(-1),
			  Xcons.IntConstant(0));
      } else {
	Xobject off = info.getAlignSubscriptOffset();
	if(off == null) off = Xcons.IntConstant(0);
	args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			  lower, upper, // modified
			  Xcons.IntConstant(info.getAlignSubscriptIndex()),
			  off);
      }
      st.insert(f.callSubroutine(args));
    }

    if(hasShadow()){
      f = env.declInternIdent(XMP.array_init_shadow_f,Xtype.FsubroutineType, block);
      for(int i = 0; i < dims.size(); i++){
	if(hasShadow(i)){
	  int left = getShadowLeft(i);
	  int right = getShadowRight(i);
	  if(isFullShadow(i)) left = right = -1;
	  args = Xcons.List(descId.Ref(),
			    Xcons.IntConstant(i),
			    Xcons.IntConstant(left),
			    Xcons.IntConstant(right));
	  st.insert(f.callSubroutine(args));
	}
      }
    }

    f = env.declInternIdent(XMP.array_init_f,Xtype.FsubroutineType, block);
    st.insert(f.callSubroutine(Xcons.List(descId.Ref())));

    if(isLinearized()){
      // allocate size variable
      Xobject alloc_size = null;
      for(int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);
	f = env.declInternIdent(XMP.array_get_local_size_f,
			      Xtype.FsubroutineType, block);
	st.insert(f.callSubroutine(Xcons.List(descId.Ref(),
					      Xcons.IntConstant(i),
					      info.getArraySizeVar().Ref(),
					      info.getArrayOffsetVar().Ref(),
					      info.getArrayBlkOffsetVar().Ref())));
	if(alloc_size == null)
	  alloc_size = info.getArraySizeVar().Ref();
	else
	  alloc_size = Xcons.binaryOp(Xcode.MUL_EXPR, 
				      alloc_size,info.getArraySizeVar().Ref());
      }
      
      // allocatable
      Xobject size_1 = Xcons.binaryOp(Xcode.MINUS_EXPR,alloc_size,
				      Xcons.IntConstant(1));

      alloc.setArg(0, localId.Ref());
      alloc.setArg(1, Xcons.FindexRange(Xcons.IntConstant(0),size_1));
    } else {
      XobjList alloc_args = Xcons.List();
      for(int i = 0; i < dims.size(); i++){
	XMPdimInfo info = dims.elementAt(i);
	f = env.declInternIdent(XMP.array_get_local_size_f,
			      Xtype.FsubroutineType, block);
	st.insert(f.callSubroutine(Xcons.List(descId.Ref(),
					      Xcons.IntConstant(i),
					      info.getArraySizeVar().Ref(),
					      info.getArrayOffsetVar().Ref(),
					      info.getArrayBlkOffsetVar().Ref())));
	if (isDistributed(i)){
	    // distributed
	    Xobject size_1 = Xcons.binaryOp(Xcode.MINUS_EXPR,
					    info.getArraySizeVar().Ref(),
					    Xcons.IntConstant(1));
	    alloc_args.add(Xcons.FindexRange(Xcons.IntConstant(0),size_1));
	}
	else {
	    // not distributed
	    alloc_args.add(Xcons.FindexRange(orig_lower[i],
					     info.getArraySizeVar().Ref()));
	}
      }
      
      // allocatable
      alloc.setArg(0, localId.Ref());
      alloc.setArg(1, alloc_args);
    }
      
    // set
    f = env.declInternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType, block);
    Xobject isCoarray = (sclass == StorageClass.FSAVE) ? Xcons.IntConstant(1) : Xcons.IntConstant(0);
    st.add(f.callSubroutine(Xcons.List(descId.Ref(), localId.Ref(), isCoarray)));

  }

  /*
   * rewrite Deallocate for aligned arrays
   */
  public void rewriteDeallocate(XobjList dealloc, Statement st,
				Block block, XMPenv env){
    Ident f;

    // deallocate desc
    f = env.declInternIdent(XMP.array_deallocate_f, Xtype.FsubroutineType, block);
    st.insert(f.callSubroutine(Xcons.List(descId.Ref())));

    dealloc.setArg(0, localId.Ref());

  }
  
  /*
   * rewrite pointer assignment for aligned arrays
   */
  public void rewritePointerAssign(Xobject pointer_assign, XMParray rhs_array, Statement st,
				   Block block, XMPenv env){

    Ident f;
    Xobject args;

    // Following codes come from XMParray.buildConstructor

    f = env.declInternIdent(XMP.init_allocated_f, Xtype.FsubroutineType, block);
    args = Xcons.List(descId.Ref());
    st.insert(f.callSubroutine(args));

    f = env.declInternIdent(XMP.assign_align_info_f, Xtype.FsubroutineType, block);
    args = Xcons.List(descId.Ref(), rhs_array.getDescId().Ref());
    st.insert(f.callSubroutine(args));
    
    if(hasShadow()){
      f = env.declInternIdent(XMP.array_init_shadow_f,Xtype.FsubroutineType, block);
      for(int i = 0; i < dims.size(); i++){
  	if(hasShadow(i)){
  	  int left = getShadowLeft(i);
  	  int right = getShadowRight(i);
  	  if(isFullShadow(i)) left = right = -1;
  	  args = Xcons.List(descId.Ref(),
  			    Xcons.IntConstant(i),
  			    Xcons.IntConstant(left),
  			    Xcons.IntConstant(right));
  	  st.insert(f.callSubroutine(args));
  	}
      }
    }

    // Following codes come from XMParray.buildConstructor
    f = env.declInternIdent(XMP.init_allocated_f, Xtype.FsubroutineType, block);
    args = Xcons.List(descId.Ref());
    st.insert(f.callSubroutine(args));

    f = env.declInternIdent(XMP.array_align_info_f,Xtype.FsubroutineType, block);

    // translate the statement (lower bounds to be zero if the axis is distributed)
    XobjList spec = Xcons.List();
    for (int i = 0; i < dims.size(); i++){
      if (isDistributed(i)){
    	// distributed
    	spec.add(Xcons.FindexRangeOfAssumedShape(Xcons.IntConstant(0)));
      }
      else {
    	// not distributed
    	spec.add(Xcons.FindexRangeOfAssumedShape(this.getLowerAt(i)));
      }
    }
      
    pointer_assign.setArg(0, Xcons.FarrayRef(localId.Ref(), spec));
    //pointer_assign.setArg(1, rhs_array.getLocalId().Ref());
      
    f = env.declInternIdent(XMP.array_init_f,Xtype.FsubroutineType, block);
    st.insert(f.callSubroutine(Xcons.List(descId.Ref())));
      
    // set
    f = env.declInternIdent(XMP.array_set_local_array_f,Xtype.FsubroutineType, block);
    Xobject isCoarray = (sclass == StorageClass.FSAVE) ? Xcons.IntConstant(1) : Xcons.IntConstant(0);
    st.add(f.callSubroutine(Xcons.List(descId.Ref(), localId.Ref(), isCoarray)));

  }

  public void buildSetup(BlockList body, XMPenv env){
    Ident f;
    Xobject args;
    
    // not yet
  }

  public void buildDestructor(BlockList body, XMPenv env, Block block){
    if (!is_saveDesc){
      Ident f;
      Xobject args;
    
      f = env.declInternIdent(XMP.array_dealloc_f,Xtype.FsubroutineType, block);
      args = Xcons.List(descId.Ref());
      body.insert(f.callSubroutine(args));
    }
  }

  public Xobject convertOffset(int dim_i){
      XMPdimInfo info = dims.elementAt(dim_i);
      if(!isDistributed(dim_i)){  // case not aligned, duplicated
	  //return info.getLower();
	  if (is_linearized) return info.getLower();
	  else return null;
      }
      Xobject offset = info.getAlignSubscriptOffset();
      Xobject alb = info.getLower();
      Xobject tlb = this.getAlignTemplate().getLowerAt(info.getAlignSubscriptIndex());
      if ((offset != null && !offset.isZeroConstant()) || !alb.equals(tlb) || !getAlignTemplate().isFixed())
	  return info.getArrayOffsetVar().Ref();
      else
	  return null;
  }

  public Xobject conertSize(int dim_i){
      XMPdimInfo info = dims.elementAt(dim_i);
      return info.getArraySizeVar().Ref();
  }

  public Xobject convertLinearIndex(Xobject index_list){
    Xobject idx = null;
    for(int i = dims.size()-1; i >= 0; i--){
      XMPdimInfo info = dims.elementAt(i);
      Xobject x = index_list.getArg(i);
      if(x.Opcode() != Xcode.F_ARRAY_INDEX)
	 XMP.fatal("convertLinearIndex: not F_ARRAY_INDEX");
      x = x.getArg(0);

      if(idx == null) idx = x;
      else idx = Xcons.binaryOp(Xcode.PLUS_EXPR,idx,x);
      if(i != 0) {
	Xobject size = dims.elementAt(i-1).getArraySizeVar().Ref();
	idx = Xcons.binaryOp(Xcode.MUL_EXPR,idx,size);
      }
    }
    return Xcons.List(Xcons.FarrayIndex(idx));
  }
}
