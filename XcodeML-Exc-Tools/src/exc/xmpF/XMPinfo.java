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
  Xobject async_id;
  Vector<Xobject> waitAsyncIds;

  // loop info for loop
  Vector<XMPdimInfo> loop_dims;  // and on_ref

  // for reflect
  Vector<XMParray> reflectArrays; // and async_id
  Vector<XMPdimInfo> widthList;

  // for reduction
  int reduction_op; 
  Vector<Ident> reduction_vars; 
  Vector<Vector<Ident>> reduction_pos_vars;

  // for bcast
  XMPobjectsRef bcast_from; // and on_ref, info_vars

  // for gmove
  Xobject gmoveLeft,gmoveRight;
  Xobject gmoveOpt;

  // for template_fix
  XMPtemplate template;
  XobjList sizeList;
  XobjList distList;

  // for task
  boolean nocomm_flag;

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

  public void setAsyncId(Xobject async_id) { this.async_id = async_id; }

  public Xobject getAsyncId() { return async_id; }

  public void setWaitAsyncIds(Vector<Xobject> waitAsyncIds){
      this.waitAsyncIds = waitAsyncIds;
  }

  public Vector<Xobject> getWaitAsyncIds() { return waitAsyncIds; }

  /* 
   * for loop
   */
  public void setLoopInfo(Vector<XMPdimInfo> dims, XMPobjectsRef ref){
    loop_dims = dims;
    on_ref = ref;
  }

  public int getLoopDim() { return loop_dims.size(); }
  
  public XMPdimInfo getLoopDimInfo(int i) { return loop_dims.elementAt(i); }

  public Xobject getLoopVar(int i) { 
    return loop_dims.elementAt(i).getLoopVar(); 
  }
  
  public void setReflectArrays(Vector<XMParray> arrays){
    reflectArrays = arrays;
  }

  public void setReflectArrays(Vector<XMParray> arrays, Vector<XMPdimInfo> list){
    reflectArrays = arrays;
    widthList = list;
  }
  
  public  Vector<XMParray> getReflectArrays(){ return reflectArrays; }

  public  Vector<XMPdimInfo> getWidthList() {
      return widthList;
  }

  public void setReductionInfo(int op, Vector<Ident> vars, Vector<Vector<Ident>> pos_vars){
    reduction_op = op;
    reduction_vars = vars;
    reduction_pos_vars = pos_vars;
  }

  public void setBcastInfo(XMPobjectsRef from, XMPobjectsRef on,
			   Vector<Ident> vars){
    bcast_from = from;
    on_ref = on;
    info_vars = vars;
  }

  public int getReductionOp() { return reduction_op; }
  public Vector<Ident> getReductionVars() { return reduction_vars; }
  public Vector<Vector<Ident>> getReductionPosVars() { return reduction_pos_vars; }

  public XMPobjectsRef getBcastFrom() { return bcast_from; }

  /* for gmove */
  public void setGmoveOperands(Xobject left, Xobject right){
    gmoveLeft = left;
    gmoveRight = right;
  }

  public Xobject getGmoveLeft() { return gmoveLeft; }
  
  public Xobject getGmoveRight() { return gmoveRight; }

  public void setGmoveOpt(Xobject _gmoveOpt){
    gmoveOpt = _gmoveOpt;
  }

  public Xobject getGmoveOpt() { return gmoveOpt; }

  public void setTemplateFix(XMPtemplate t, XobjList sList, XobjList dList){
    template = t;
    sizeList = sList;
    distList = dList;
  }

  public XMPtemplate getTemplate() { return template; }
  public XobjList getSizeList() { return sizeList; }
  public XobjList getDistList() { return distList; }

  public void setNocomm(Xobject nocomm){
    nocomm_flag = (nocomm.getInt() == 1);
  }

  public boolean isNocomm() { return nocomm_flag; }
}
