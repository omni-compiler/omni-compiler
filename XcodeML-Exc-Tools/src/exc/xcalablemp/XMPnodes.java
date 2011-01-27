/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPnodes extends XMPobject {
  public final static int INHERIT_GLOBAL	= 10;
  public final static int INHERIT_EXEC		= 11;
  public final static int INHERIT_NODES		= 12;

  public final static int MAP_UNDEFINED		= 20;
  public final static int MAP_REGULAR		= 21;

  public XMPnodes(String name, int dim, Ident descId) {
    super(XMPobject.NODES, name, dim, descId);
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

    // check <map-type> := { <undefined> | regular }
    int nodesMapType = 0;
    if (nodesDecl.getArg(0) == null) {
      nodesMapType = XMPnodes.MAP_UNDEFINED;
    }
    else {
      XMP.warning("'regular' is not supported in this version");
      nodesMapType = XMPnodes.MAP_REGULAR;
    }

    // check name collision
    String nodesName = nodesDecl.getArg(1).getString();
    if (isLocalPragma) {
      XMPlocalDecl.checkObjectNameCollision(nodesName, funcBlockList, localXMPsymbolTable);
    }
    else {
      globalDecl.checkObjectNameCollision(nodesName);
    }

    // declare nodes desciptor
    Ident nodesDescId = null;
    if (isLocalPragma) {
      nodesDescId = XMPlocalDecl.addObjectId(XMP.DESC_PREFIX_ + nodesName, pb);
    }
    else {
      nodesDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + nodesName, Xtype.voidPtrType);
    }

    // declare nodes object
    int nodesDim = 0;
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) nodesDim++;
    if (nodesDim > XMP.MAX_DIM) {
      throw new XMPexception("nodes dimension should be less than " + (XMP.MAX_DIM + 1));
    }

    XMPnodes nodesObject = new XMPnodes(nodesName, nodesDim, nodesDescId);
    if (isLocalPragma) {
      localXMPsymbolTable.putXMPobject(nodesObject);
    }
    else {
      globalDecl.putXMPobject(nodesObject);
    }

    // create function call
    XobjList nodesArgs = Xcons.List(Xcons.IntConstant(nodesMapType), nodesDescId.getAddr(), Xcons.IntConstant(nodesDim));

    XobjList inheritDecl = (XobjList)nodesDecl.getArg(3);
    String inheritType = null;
    String nodesRefType = null;
    XobjList nodesRef = null;
    XMPnodes nodesRefObject = null;
    switch (inheritDecl.getArg(0).getInt()) {
      case XMPnodes.INHERIT_GLOBAL:
        inheritType = "GLOBAL";
        break;
      case XMPnodes.INHERIT_EXEC:
        inheritType = "EXEC";
        break;
      case XMPnodes.INHERIT_NODES:
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
            nodesRefType = "NAMED";

            String nodesRefName = nodesRef.getArg(0).getString();

            if (isLocalPragma) {
              nodesRefObject = XMPlocalDecl.getXMPnodes(nodesRefName, localXMPsymbolTable, globalDecl);
            }
            else {
              nodesRefObject = globalDecl.getXMPnodes(nodesRefName);
            }

            if (nodesRefObject == null) {
              throw new XMPexception("cannot find nodes '" + nodesRefName + "'");
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
    for (XobjArgs i = nodesDecl.getArg(2).getArgs(); i != null; i = i.nextArgs()) {
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
      XMPlocalDecl.addConstructorCall(initFuncName, nodesArgs, pb, globalDecl);
      XMPlocalDecl.insertDestructorCall("_XMP_finalize_nodes", Xcons.List(nodesDescId.Ref()), pb, globalDecl);
    }
    else {
      globalDecl.addGlobalInitFuncCall(initFuncName, nodesArgs);
    }
  }
}
