/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPnodes extends XMPobject {
  public final static int INHERIT_NULL		= -1;
  public final static int INHERIT_GLOBAL	= 10;
  public final static int INHERIT_EXEC		= 11;
  public final static int INHERIT_NODES		= 12;

  private int _inheritType;
  private boolean _inheritNamedNodes;
  private XMPnodes _inheritNodes;

  public XMPnodes(String name, int dim, Ident descId) {
    super(XMPobject.NODES, name, dim, descId);
    _inheritType = INHERIT_NULL;
    _inheritNamedNodes = false;
    _inheritNodes = null;
  }

  private int getInheritType() {
    return _inheritType;
  }

  private void setInheritType(int type) {
    _inheritType = type;
  }
  
  private boolean checkInheritNamedNodes() {
    return _inheritNamedNodes;
  }

  private void setInheritNamedNodes() {
    _inheritNamedNodes = true;
  }

  private XMPnodes getInheritNodes() {
    return _inheritNodes;
  }

  private void setInheritNodes(XMPnodes nodes) {
    _inheritNodes = nodes;
  }

  public static void translateNodes(XobjList nodesDecl, XMPglobalDecl globalDecl,
                                    boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // local parameters
    BlockList funcBlockList = null;
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      funcBlockList = XMPlocalDecl.findParentFunctionBlock(pb).getBody();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // check name collision
    String nodesName = nodesDecl.getArg(0).getString();
    if (isLocalPragma) {
      XMPlocalDecl.checkObjectNameCollision(nodesName, funcBlockList, localXMPsymbolTable);
    } else {
      globalDecl.checkObjectNameCollision(nodesName);
    }

    // declare nodes desciptor
    Ident nodesDescId = null;
    if (isLocalPragma) {
      nodesDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + nodesName, pb);
    } else {
      nodesDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + nodesName, Xtype.voidPtrType);
    }

    // declare nodes object
    int nodesDim = 0;
    for (XobjArgs i = nodesDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) nodesDim++;
    if (nodesDim > XMP.MAX_DIM) {
      throw new XMPexception("nodes dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XMPnodes nodesObject = new XMPnodes(nodesName, nodesDim, nodesDescId);
    if (isLocalPragma) {
      localXMPsymbolTable.putXMPobject(nodesObject);
    } else {
      globalDecl.putXMPobject(nodesObject);
    }

    // create function call
    XobjList nodesArgs = Xcons.List(nodesDescId.getAddr(), Xcons.IntConstant(nodesDim));

    XobjList inheritDecl = (XobjList)nodesDecl.getArg(2);
    nodesObject.setInheritType(inheritDecl.getArg(0).getInt());
    String inheritType = null;
    String nodesRefType = null;
    XobjList nodesRef = null;
    switch (nodesObject.getInheritType()) {
      case INHERIT_GLOBAL:
        inheritType = "GLOBAL";
        break;
      case INHERIT_EXEC:
        inheritType = "EXEC";
        break;
      case INHERIT_NODES:
        {
          inheritType = "NODES";

          nodesRef = (XobjList)inheritDecl.getArg(1);
          if (nodesRef.getArg(0) == null) {
            nodesRefType = "NUMBER";

            XobjList nodeNumberTriplet = (XobjList)nodesRef.getArg(1);
            // lower
            if (nodeNumberTriplet.getArg(0) == null) nodesArgs.add(Xcons.IntConstant(1));
            else nodesArgs.add(nodeNumberTriplet.getArg(0));
            // upper
            if (nodeNumberTriplet.getArg(1) == null) nodesArgs.add(globalDecl.getWorldSizeId().Ref());
            else nodesArgs.add(nodeNumberTriplet.getArg(1));
            // stride
            if (nodeNumberTriplet.getArg(2) == null) nodesArgs.add(Xcons.IntConstant(1));
            else nodesArgs.add(nodeNumberTriplet.getArg(2));
          }
          else {
            XMPnodes nodesRefObject = null;
            nodesRefType = "NAMED";
            nodesObject.setInheritNamedNodes();

            String nodesRefName = nodesRef.getArg(0).getString();

            if (isLocalPragma) {
              nodesRefObject = XMPlocalDecl.getXMPnodes(nodesRefName, localXMPsymbolTable, globalDecl);
            } else {
              nodesRefObject = globalDecl.getXMPnodes(nodesRefName);
            }

            if (nodesRefObject == null) {
              throw new XMPexception("cannot find nodes '" + nodesRefName + "'");
            } else {
              nodesObject.setInheritNodes(nodesRefObject);
            }

            nodesArgs.add(nodesRefObject.getDescId().Ref());

            int nodesRefDim = nodesRefObject.getDim();
            boolean isDynamicNodesRef = false;
            XobjList subscriptList = (XobjList)nodesRef.getArg(1);
            if (subscriptList == null) {
              for (int nodesRefIndex = 0; nodesRefIndex < nodesRefDim; nodesRefIndex++) {
                // lower
                nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                // upper
                Xobject nodesRefSize = nodesRefObject.getUpperAt(nodesRefIndex);
                if (nodesRefSize == null) isDynamicNodesRef = true;
                else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesRefSize));
                // stride
                nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
              }
            }
            else {
              int nodesRefIndex = 0;
              for (XobjArgs i = subscriptList.getArgs(); i != null; i = i.nextArgs()) {
                if (nodesRefIndex == nodesRefDim)
                  throw new XMPexception("wrong nodes dimension indicated, too many");

                XobjList subscriptTriplet = (XobjList)i.getArg();
                // lower
                if (subscriptTriplet.getArg(0) == null) nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(0)));
                // upper
                if (subscriptTriplet.getArg(1) == null) {
                  Xobject nodesRefSize = nodesRefObject.getUpperAt(nodesRefIndex);
                  if (nodesRefSize == null) isDynamicNodesRef = true;
                  else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesRefSize));
                }
                else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(1)));
                // stride
                if (subscriptTriplet.getArg(2) == null) nodesArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
                else nodesArgs.add(Xcons.Cast(Xtype.intType, subscriptTriplet.getArg(2)));

                nodesRefIndex++;
              }

              if (nodesRefIndex != nodesRefDim)
                throw new XMPexception("the number of <nodes-subscript> should be the same with the nodes dimension");
            }

            if (isDynamicNodesRef) nodesArgs.cons(Xcons.IntConstant(1));
            else                   nodesArgs.cons(Xcons.IntConstant(0));
          }
          break;
        }
      default:
        throw new XMPexception("cannot create sub node set, unknown operation in nodes directive");
    }

    boolean isDynamic = false;
    for (XobjArgs i = nodesDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      Xobject nodesSize = i.getArg();
      if (nodesSize == null) isDynamic = true;
      else nodesArgs.add(Xcons.Cast(Xtype.intType, nodesSize));

      nodesObject.addUpper(nodesSize);
    }

    String allocType = null;
    if (isDynamic)      allocType = "DYNAMIC";
    else                allocType = "STATIC";

    // add constructor call
    String initFuncName = null;
    if (nodesRef == null) {
      initFuncName = new String("_XMP_init_nodes_" + allocType + "_" + inheritType);
    }
    else {
      initFuncName = new String("_XMP_init_nodes_" + allocType + "_" + inheritType + "_" + nodesRefType);
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall(initFuncName, nodesArgs, globalDecl, pb);
      XMPlocalDecl.insertDestructorCall("_XMP_finalize_nodes", Xcons.List(nodesDescId.Ref()), globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall(initFuncName, nodesArgs);
    }
  }
}
