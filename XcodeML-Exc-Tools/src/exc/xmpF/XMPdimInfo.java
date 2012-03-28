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
  Xobject align_subscript_expr;

  final static int ALIGN_NONE = 0;
  final static int ALIGN_SET = 1;
  final static int ALIGN_ANY = 2; /* for '*' */

  // shadow for array
  int shadow_status;
  Xobject shadow_left, shadow_right;
  
  public final static int SHADOW_NONE	= 0;
  public final static int SHADOW_SET	= 1;
  public final static int SHADOW_FULL	= 2;

  // for loop
  Xobject loop_var;
  ForBlock loop_block;

  // on_ref for loop
  int on_ref_index;
  Xobject on_ref_offset;


  // null constructor
  public XMPdimInfo() { 
    distManner = 0;
    align_status = ALIGN_NONE;
    lower = null;
    upper = null;
    stride = null;
    on_ref_index = -1;
  }  
  
  public Xobject getLower() { 
    if(lower == null) return Xcons.IntConstant(1);
    return lower; 
  }
  
  public Xobject getUpper() { return upper; }

  public Xobject getSize() { 
    if(lower == null) return upper;
    else return Xcons.binaryOp(Xcode.PLUS_EXPR,upper,lower); 
  }

  public Xobject getStride() { 
    if(stride == null) return Xcons.IntConstant(1);
    return stride; 
  }

  public boolean isStar() { return is_star; }

  public Xobject getIndex() { return upper; }

  public boolean isTriplet() {
    return (lower != null || stride != null);
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
    align_subscript_expr = expr;
  }

  public void setAlignAny() { align_status = ALIGN_ANY; }

  public int getAlignSubscriptIndex() { return align_subscript_index; }
  
  public Xobject getAlignSubscriptExpr() { return align_subscript_expr; }
  
  public boolean isAlignAny() { return  align_status == ALIGN_ANY; }


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
    } else {
      upper = decl.getArg(0);
      lower = decl.getArg(1);
      stride = decl.getArg(2);
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

  public void setLoopOnRefInfo(int index, Xobject offset){
    on_ref_index = index;
    on_ref_offset = offset;
  }

  public int getLoopOnRefIndex() { return on_ref_index; }
  
  public Xobject getLoopOnRefOffset() { return  on_ref_offset; }

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
	s += ",align("+align_subscript_index+","+align_subscript_expr+")";
      }
    }
    s += ">";
    return s;
  }
}
