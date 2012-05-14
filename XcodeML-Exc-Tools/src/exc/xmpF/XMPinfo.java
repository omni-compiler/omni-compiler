/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.Vector;

/**
 * information for each XMP directive
 */
public class XMPinfo
{
  private XMPinfo parent;
  private Block block; /* back link */
  XMPpragma pragma; /* directives */
  XMPenv env;

  BlockList body;

  XMPobjectsRef on_ref;
  Vector<Ident> info_vars; 

  // loop info for loop
  Vector<XMPdimInfo> loop_dims;  // and on_ref

  // for reflect
  Vector<XMParray> reflectArrays; // and on_ref

  // for reduction
  int reduction_op;  // and info_vars

  // for bcast
  XMPobjectsRef bcast_from; // and on_ref, info_vars

  public XMPinfo(XMPpragma pragma, XMPinfo parent, Block b, XMPenv env) {
    this.pragma = pragma;
    this.parent = parent;
    this.block = b;
    this.env = env;
  }
    
  public Block getBlock()  {
    return block;
  }

  public void setBody(BlockList body) { this.body = body; }
  
  public BlockList getBody() { return body; }

  public void setOnRef(XMPobjectsRef ref) { on_ref = ref; }

  public XMPobjectsRef getOnRef() { return on_ref; }

  public Vector<Ident> getInfoVarIdents() { return info_vars; }

  /* 
   * for loop
   */
  public void setLoopInfo(Vector<XMPdimInfo> dims, XMPobjectsRef ref,
			  Xobject reduction_ref){
    loop_dims = dims;
    on_ref = ref;
    // loop_reduction_ref = reduction_ref;
  }

  public int getLoopDim() { return loop_dims.size(); }
  
  public XMPdimInfo getLoopDimInfo(int i) { return loop_dims.elementAt(i); }

  public Xobject getLoopVar(int i) { 
    return loop_dims.elementAt(i).getLoopVar(); 
  }
  
  // public Xobject getLoopReductionRef() {  return loop_reduction_ref; }
  
  public void setReflectArrays(Vector<XMParray> arrays){
    reflectArrays = arrays;
  }
  
  public  Vector<XMParray> getReflectArrays(){ return reflectArrays; }

  public void setReductionInfo(int op, Vector<Ident> vars){
    reduction_op = op;
    info_vars = vars;
  }

  public void setBcastInfo(XMPobjectsRef from, XMPobjectsRef on,
			   Vector<Ident> vars){
    bcast_from = from;
    on_ref = on;
    info_vars = vars;
  }

  public int getReductionOp() { return reduction_op; }

  public XMPobjectsRef getBcastFrom() { return bcast_from; }

}
