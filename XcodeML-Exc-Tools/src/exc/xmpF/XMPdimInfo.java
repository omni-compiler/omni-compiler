/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 * $
 */

package exc.xmpF;

import exc.block.*;
import exc.object.*;
import java.util.Vector;

public class XMPdimInfo {

  Xobject lower, upper, stride;
  boolean is_star = false;
  
  // distribution for template
  int distManner;
  Xobject distArg;

  // align for array
  int align_status;
  int align_subscript_index;
  Xobject align_subscript_offset;
  Ident a_dim_size_var; 
  Ident a_offset_var;
  Ident a_blk_offset_var;

  final static int ALIGN_NONE = 0;
  final static int ALIGN_SET = 1;
  final static int ALIGN_ANY = 2; /* for '*' */

  // shadow for array
  boolean is_full_shadow;
  int shadow_left;
  int shadow_right;

  // for loop
  Xobject loop_var;
  ForBlock loop_block;
  Ident loop_local_var;
  int loop_on_index;

  // on_ref for loop
  int on_ref_loop_index;
  Xobject on_ref_offset;

  // null constructor
  public XMPdimInfo() { 
    distManner = 0;
    align_status = ALIGN_NONE;
    lower = null;
    upper = null;
    stride = null;

    loop_on_index = -1;
    on_ref_loop_index = -1;

    is_full_shadow = false;
    shadow_left  = 0;
    shadow_right = 0;
  }  
  
  public Xobject getLower() { 
    if(lower == null) return Xcons.IntConstant(1);
    return lower; 
  }

  //public Xobject getLower() { return lower; }

  public void setLower(Xobject l) {
    lower = l;
  }

  public boolean hasLower(){
    return (lower != null);
  }
  
  public Xobject getUpper() { return upper; }

  public void setUpper(Xobject u) {
    upper = u;
  }

  public boolean hasUpper(){
    return (upper != null);
  }

  public Xobject getSize() { 
    if(lower == null) return upper;
    else return Xcons.binaryOp(Xcode.PLUS_EXPR,upper,lower); 
  }

  public boolean hasStride() { 
    return stride != null;
  }

  public Xobject getStride() { 
    if(stride == null) return Xcons.IntConstant(1);
    return stride; 
  }

  public void setStride(Xobject s) {
    stride = s;
  }

  public boolean isStar() { return is_star; }

  public void setStar() { is_star = true; }

  public boolean isScalar(){
    return (stride != null && stride.isZeroConstant());
  }

  public Xobject getIndex() { return upper; }

  public boolean isTriplet() {
    return (!is_star && !isScalar());
    //    return (lower != null || stride != null);
  }

  /*
   * distribution for template
   */
  public void setDistManner(int distManner, Xobject distArg){
    this.distManner = distManner;
    this.distArg = distArg;
  }

  public int getDistManner() { return distManner; }
  
  public Xobject getDistArg() { return distArg; }

  /*
   *  align for array
   */
  public void setAlignSubscript(int idx, Xobject expr){
    align_status = ALIGN_SET;
    align_subscript_index = idx;
    align_subscript_offset = expr;
  }

  public void setAlignAny() { align_status = ALIGN_ANY; }

  public int getAlignSubscriptIndex() { return align_subscript_index; }
  
  public Xobject getAlignSubscriptOffset() { return align_subscript_offset; }
  
  public boolean isAlignAny() { return  align_status == ALIGN_ANY; }

  public void setArrayInfoVar(Ident size_var, Ident off_var, Ident blk_off_var){
    a_dim_size_var = size_var;
    a_offset_var = off_var;
    a_blk_offset_var = blk_off_var;
  }

  public Ident getArraySizeVar() { return a_dim_size_var; }
  
  public Ident getArrayOffsetVar() { return a_offset_var; }

  public Ident getArrayBlkOffsetVar() { return a_blk_offset_var; }

  /*
   * parse dim expression
   */
  static XMPdimInfo parseDecl(Xobject decl){
    XMPdimInfo t = new XMPdimInfo();
    t.parse(decl);
    return t;
  }
  
  void parse(Xobject decl){
    if(decl == null){
      is_star = true;
      return;
    } 

    if(decl.Opcode() != Xcode.LIST){
      upper = decl;
      stride = Xcons.IntConstant(0);
    } else {
      lower = decl.getArg(0);     /* null in case of ":" */
      upper = decl.getArgOrNull(1); 
      stride = decl.getArgOrNull(2);
    }
  }
  
  public static Vector<XMPdimInfo> parseSubscripts(Xobject subscriptList){
    Vector<XMPdimInfo> subscripts = new Vector<XMPdimInfo>();
    for (XobjArgs i = subscriptList.getArgs(); i != null; i = i.nextArgs()) 
      subscripts.add(XMPdimInfo.parseDecl(i.getArg()));
    return subscripts;
  }

  public static XMPdimInfo createFromRange(Xobject x){
    XMPdimInfo i = new XMPdimInfo();
    if(x.Opcode() != Xcode.F_INDEX_RANGE){
      XMP.fatal("XMPdimInfo: createFromRage, not F_INDEX_RANGE x="+x);
    }
    i.lower = x.getArg(0);
    i.upper = x.getArg(1);
    i.stride = x.getArg(2);
    return i;
  }

  /*
   * For loop
   */
  public static XMPdimInfo loopInfo(ForBlock block){
    XMPdimInfo i = new XMPdimInfo();
    i.loop_block = block;
    i.loop_var = block.getInductionVar();
    i.upper = block.getUpperBound();
    i.lower = block.getLowerBound();
    i.stride =block.getStep();
    return i;
  }

  public Xobject getLoopVar() { return loop_var; }
  
  public ForBlock getLoopBlock() { return loop_block; }

  public void setLoopLocalVar(Ident id) { loop_local_var = id; }

  public Ident getLoopLocalVar() { return loop_local_var; }

  public void setLoopOnIndex(int index) { loop_on_index = index; }
  
  public int getLoopOnIndex() { return loop_on_index; }

  /*
   * For on_ref
   */
  public void setLoopOnRefInfo(int index, Xobject offset){
    on_ref_loop_index = index;
    on_ref_offset = offset;
  }

  public int getOnRefLoopIndex() { return on_ref_loop_index; }
  
  public Xobject getOnRefOffset() { return  on_ref_offset; }

  public String toString(){
    String s = "<";
    if(lower != null) s += lower;
    s += ":";
    if(upper != null) s += upper;
    if(stride != null){
      s += ",stride=";
      s += stride;
    }
    if(distManner != 0){
      s += ",dist("+XMPtemplate.distMannerName(distManner)+")=";
      if(distArg != null) s += distArg;
    }
    if(align_status != ALIGN_NONE){
      if(align_status == ALIGN_ANY){
	s += ",align(*)";
      } else {
	s += ",align("+align_subscript_index+","+align_subscript_offset+")";
      }
    }
    s += ">";
    return s;
  }
}
