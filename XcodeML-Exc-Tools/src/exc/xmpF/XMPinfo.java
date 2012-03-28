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

  // loop info for loop
  Vector<XMPdimInfo> loop_dims;
  XMPobjectsRef loop_on_ref;
  Xobject loop_reduction_ref;
  
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

  public void setLoopInfo(Vector<XMPdimInfo> dims, XMPobjectsRef on_ref,
			  Xobject reduction_ref){
    loop_dims = dims;
    loop_on_ref = on_ref;
    loop_reduction_ref = reduction_ref;
  }

  public int getLoopDim() { return loop_dims.size(); }
  
  public Xobject getLoopIndex(int i) { return loop_dims.elementAt(i).getIndex(); }

  public XMPobjectsRef getLoopOnRef() { return loop_on_ref; }
  
  public Xobject getLoopReductionRef() {  return loop_reduction_ref; }
  
}
