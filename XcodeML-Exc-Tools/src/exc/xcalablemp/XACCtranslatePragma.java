package exc.xcalablemp;

import java.util.Iterator;

import exc.block.*;
import exc.object.*;
import exc.openacc.ACCpragma;

public class XACCtranslatePragma {
  //XACCsymbolTable _globalSymbolTable;
  private XMPglobalDecl _globalDecl;
  //private XACCglobalDecl _xaccGlobalDecl;
  static final String XACC_DESC_PREFIX = "_XACC_DESC_";
  
  public XACCtranslatePragma(XMPglobalDecl globalDecl){
    //_globalSymbolTable = new XACCsymbolTable();
    _globalDecl = globalDecl;
    //_xaccGlobalDecl = new XACCglobalDecl(globalDecl.getEnv());
  }

  public void translate(Xobject x){
    String pragmaName = x.getArg(0).getName();
    ACCpragma pragma = ACCpragma.valueOf(pragmaName);

    try{
      switch(pragma){
      case DECLARE:
        translateDeclare(x); break;
      case DEVICE:
        translateDevice(x); break;
      default:
        XMP.fatal("unimplemented xacc pragma :" + pragma.getName());
      }
    } catch (XMPexception e) {
      XMP.error(x.getLineNo(), e.getMessage());
    }
  }
  
  private void translateDevice(Xobject x) throws XMPexception{
    XobjList deviceDecl = (XobjList)x.getArg(1);
    XobjList deviceDeclCopy = (XobjList)deviceDecl.copy();
    XMPdevice.translateDevice(deviceDeclCopy, _globalDecl, false, null);
    
    //remove pragma
    XobjectDef def = (XobjectDef)x.getParent();
    Iterator<XobjectDef> iter = _globalDecl.getEnv().iterator();
    while(iter.hasNext()){
      XobjectDef d = iter.next();
      if(def == d){
        iter.remove();
      }
    }
  }
  
  private void translateDeclare(Xobject x){
    //System.out.println("trans decl");
    XobjList clauses = (XobjList)x.getArg(1);
    ACCpragma pragma = ACCpragma.DECLARE;
    
    BlockList newBody = Bcons.emptyBody();
    
    XMPdevice device = null;
    XMPlayout layout = null;
    XMPon on = null;
    if (clauses != null){
      device = getXACCdevice(clauses);
      layout = getXACClayout(clauses);
    }
    
    if(XMP.XACC){
      Ident fid = _globalDecl.declExternFunc("acc_set_device_num");
      
      //base
      BlockList baseBody = Bcons.emptyBody();
      BlockList baseDeviceLoopBody = Bcons.emptyBody();
      Ident baseDeviceLoopVarId = baseBody.declLocalIdent("_XACC_device_" + device.getName(), Xtype.intType);
      baseBody.add(Bcons.FORall(baseDeviceLoopVarId.Ref(), device.getLower(), device.getUpper(),
          device.getStride(), Xcode.LOG_LE_EXPR, baseDeviceLoopBody));
      baseDeviceLoopBody.add(fid.Call(Xcons.List(baseDeviceLoopVarId.Ref(), device.getDeviceRef())));
      rewriteACCClauses(clauses, null, newBody, baseDeviceLoopBody, baseDeviceLoopVarId, device, layout);
      

      baseDeviceLoopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.DECLARE.toString(), clauses, null));
      
      BlockList beginBody = baseBody.copy();
      BlockList endBody = baseBody.copy();

      rewriteXACCPragmaData(beginBody, true);
      rewriteXACCPragmaData(endBody, false);

      _globalDecl.addXACCconstructor(beginBody.toXobject());
      _globalDecl.addXACCdestructor(endBody.toXobject());
      
//      Ident initFuncId = _globalDecl.declGlobalIdent("_XACC_init", Xtype.Function(Xtype.voidType));
//      XobjectDef initFuncDef = XobjectDef.Func(initFuncId, Xcons.List(), null, beginBody.toXobject());
//      _globalDecl.getEnv().add(initFuncDef);
//      _globalDecl.addGlobalInitFuncCall("_XACC_init", Xcons.List());
//      
//      Ident finalizeFuncId = _globalDecl.declGlobalIdent("_XACC_finalize", Xtype.Function(Xtype.voidType));
//      XobjectDef finalizeFuncDef = XobjectDef.Func(finalizeFuncId, Xcons.List(), null, endBody.toXobject());
//      _globalDecl.getEnv().add(finalizeFuncDef);
//      _globalDecl.addGlobalFinalizeFuncCall("_XACC_finalize", Xcons.List());
    }else{
      //do anything
    }
    x.setArg(1, Xcons.List());
    
    //System.out.print("trans decl end");
  }
  
  private XMPdevice getXACCdevice(XobjList clauses){
    XMPdevice onDevice = null;
    
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject x = arg.getArg();
      if(x == null) continue;
      String clauseName = x.left().getString();
      ACCpragma accClause = ACCpragma.valueOf(clauseName);
      //if(accClause == null) continue;
      if(accClause == ACCpragma.ON_DEVICE){
        String deviceName = x.right().getString();
        onDevice = (XMPdevice)_globalDecl.getXMPobject(deviceName);
        if (onDevice == null) XMP.error(x.getLineNo(), "wrong device in on_device");
        arg.setArg(null);
      }
    }
    return onDevice;
  }
  
  public void rewriteACCpragma(FunctionBlock fb, XMPsymbolTable localXMPsymbolTable){
    topdownBlockIterator bIter = new topdownBlockIterator(fb);

    for (bIter.init(); !bIter.end(); bIter.next()){
      Block block = bIter.getBlock();
      if (block.Opcode() == Xcode.ACC_PRAGMA){
        PragmaBlock pb = (PragmaBlock)block;
        Xobject clauses = pb.getClauses();
        
        ACCpragma pragma = ACCpragma.valueOf(((PragmaBlock)block).getPragma());
        BlockList newBody = Bcons.emptyBody();
        XMPdevice device = null;
        XMPlayout layout = null;
        XMPon on = null;
        if (clauses != null){
          device = getXACCdevice((XobjList)clauses, fb);
          layout = getXACClayout((XobjList)clauses);
          on = getXACCon((XobjList)clauses, block);

          if(!newBody.isEmpty() && !XMP.XACC){
            bIter.setBlock(Bcons.COMPOUND(newBody));
            newBody.add(Bcons.COMPOUND(Bcons.blockList(block))); //newBody.add(block);
          }
        }
        

        if(XMP.XACC && device != null){
          if(pragma == ACCpragma.DATA || pragma == ACCpragma.PARALLEL_LOOP || pragma == ACCpragma.WAIT){
            Ident fid = _globalDecl.declExternFunc("acc_set_device_num");
            
            //base
            BlockList baseBody = Bcons.emptyBody();
            BlockList baseDeviceLoopBody = Bcons.emptyBody();
            Ident baseDeviceLoopVarId = baseBody.declLocalIdent("_XACC_device_" + device.getName(), Xtype.intType);
            baseBody.add(Bcons.FORall(baseDeviceLoopVarId.Ref(), device.getLower(), device.getUpper(),
                device.getStride(), Xcode.LOG_LE_EXPR, baseDeviceLoopBody));
            baseDeviceLoopBody.add(fid.Call(Xcons.List(baseDeviceLoopVarId.Ref(), device.getDeviceRef())));
            rewriteACCClauses(clauses, pb, newBody, baseDeviceLoopBody, baseDeviceLoopVarId, device, layout);
            BlockList pbBody;
            if(pragma == ACCpragma.DATA || pragma == ACCpragma.WAIT){
              pbBody = null;
            }else{
              pbBody = pb.getBody();
            }
            baseDeviceLoopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, pb.getPragma(), clauses, pbBody));
            
            if(pragma == ACCpragma.DATA){
              BlockList beginBody = baseBody.copy();
              BlockList endBody = baseBody.copy();

              rewriteXACCPragmaData(beginBody, true);
              rewriteXACCPragmaData(endBody, false);

 
              newBody.add(Bcons.COMPOUND(beginBody));
              newBody.add(Bcons.COMPOUND(block.getBody()));         
              newBody.add(Bcons.COMPOUND(endBody));
            }else if(pragma == ACCpragma.PARALLEL_LOOP){
              CforBlock forBlock = (CforBlock)pb.getBody().getHead();
              if(! forBlock.isCanonical()){
                                  forBlock.Canonicalize();
              }
              String loopVarName = forBlock.getInductionVar().getSym();

              try{
                int dim = on.getCorrespondingDim(loopVarName);
                if(dim >= 0){
                  XMPlayout myLayout = on.getLayout();
                  String layoutSym = XMPlayout.getDistMannerString(myLayout.getDistMannerAt(dim));
                  Ident loopInitId = baseDeviceLoopBody.declLocalIdent("_XACC_loop_init_" + loopVarName, Xtype.intType);
                  Ident loopCondId = baseDeviceLoopBody.declLocalIdent("_XACC_loop_cond_" + loopVarName, Xtype.intType);
                  Ident loopStepId = baseDeviceLoopBody.declLocalIdent("_XACC_loop_step_" + loopVarName, Xtype.intType);
                  Ident schedLoopFuncId = _globalDecl.declExternFunc("_XACC_sched_loop_layout_"+ layoutSym);
                  Xobject oldInit, oldCond, oldStep;
                  XobjList loopIter = XMPutil.getLoopIter(forBlock, loopVarName);
                  
                  //get old loop iter
                  if(loopIter != null){
                    oldInit = ((Ident)loopIter.getArg(0)).Ref();
                    oldCond = ((Ident)loopIter.getArg(1)).Ref();
                    oldStep = ((Ident)loopIter.getArg(2)).Ref();
                  }else{
                    oldInit = forBlock.getLowerBound();
                    oldCond = forBlock.getUpperBound();
                    oldStep = forBlock.getStep();
                  }
                  XobjList schedLoopFuncArgs = 
                      Xcons.List(oldInit,oldCond, oldStep,
                          loopInitId.getAddr(), loopCondId.getAddr(), loopStepId.getAddr(), 
                                                                 on.getArrayDesc().Ref(), Xcons.IntConstant(dim), baseDeviceLoopVarId.Ref());
                  baseDeviceLoopBody.insert(schedLoopFuncId.Call(schedLoopFuncArgs));
                  
                  //rewrite loop iter
                  forBlock.setLowerBound(loopInitId.Ref());
                  forBlock.setUpperBound(loopCondId.Ref());
                  forBlock.setStep(loopStepId.Ref());
                }
                newBody.add(Bcons.COMPOUND(baseBody));
              } catch (XMPexception e) {
                XMP.error(pb.getLineNo(), e.getMessage());
              }
              
            }else{ //for wait
              newBody.add(Bcons.COMPOUND(baseBody));
            }
            bIter.setBlock(Bcons.COMPOUND(newBody));
          }
          continue;
        }else{
          rewriteACCClauses(clauses, (PragmaBlock)block, newBody, null, null, null, null);      
        }

        /*
        if (XMP.XACC && onDevice != null){
          Ident fid1 = _globalDecl.declExternFunc("_XMP_set_device_num");
          if(pragma == ACCpragma.PARALLEL_LOOP){// || pragma == ACCpragma.DATA){
            BlockList deviceLoop = Bcons.emptyBody();
            Ident var = deviceLoop.declLocalIdent("_XACC_loop", Xtype.intType);
            //Ident var = deviceLoop.declLocalIdent("_XACC_loop", Xtype.intType);
            // Ident fid0 = _globalDecl.declExternFunc("xacc_get_num_current_devices",
            //                                    Xtype.intType);

            Block deviceLoopBlock = Bcons.FORall(var.Ref(), onDevice.getLower(), onDevice.getUpper(),
                onDevice.getStride(), Xcode.LOG_LE_EXPR, newBody);

            deviceLoop.add(deviceLoopBlock);
            bIter.setBlock(Bcons.COMPOUND(deviceLoop));


            newBody.insert(fid1.Call(Xcons.List(var.Ref(), onDevice.getAccDevice().Ref())));
            newBody.add(Bcons.COMPOUND(Bcons.blockList(block)));
          }else if(pragma == ACCpragma.DATA){
            Ident var = newBody.declLocalIdent("_XACC_loop", Xtype.intType);
            Block exitDataBlock;// = Bcons.emptyBlock();
            Block enterDataBlock;// = Bcons.emptyBlock();
            Xobject setDeviceCall = fid1.Call(Xcons.List(var.Ref(), onDevice.getAccDevice().Ref()));
            
            enterDataBlock = Bcons.PRAGMA(Xcode.ACC_PRAGMA, "update", clauses, null);
            exitDataBlock = Bcons.PRAGMA(Xcode.ACC_PRAGMA, "update", Xcons.List(), null);
            
            BlockList deviceLoopEnterDataBody = Bcons.emptyBody();
            deviceLoopEnterDataBody.add(setDeviceCall);
            deviceLoopEnterDataBody.add(enterDataBlock);
            
            Block deviceLoopEnterData = Bcons.FORall(var.Ref(), onDevice.getLower(), onDevice.getUpper(),
                onDevice.getStride(), Xcode.LOG_LE_EXPR, deviceLoopEnterDataBody);
            
            BlockList deviceLoopExitDataBody = Bcons.emptyBody();
            deviceLoopExitDataBody.add(setDeviceCall);
            deviceLoopExitDataBody.add(exitDataBlock);
            
            Block deviceLoopExitData = Bcons.FORall(var.Ref(), onDevice.getLower(), onDevice.getUpper(),
                onDevice.getStride(), Xcode.LOG_LE_EXPR, deviceLoopExitDataBody);
            
            //XobjList clauses = pb. 
            
            
            bIter.setBlock(Bcons.COMPOUND(newBody));
            newBody.add(deviceLoopEnterData);
            newBody.add(Bcons.COMPOUND(block.getBody()));
            newBody.add(deviceLoopExitData);
          }
        }
*/
      }
    }
  }

  private void rewriteXACCPragmaData(BlockList body, Boolean isEnter) {
    PragmaBlock pb = null;
    
    BlockIterator iter = new topdownBlockIterator(body);
    for(iter.init(); !iter.end(); iter.next()){
      Block block = iter.getBlock();
      if(block.Opcode() == Xcode.ACC_PRAGMA){
        pb = (PragmaBlock)block;
      }
    }
    
    XobjList clauses = (XobjList)pb.getClauses();
    Block newPB = Bcons.PRAGMA(Xcode.ACC_PRAGMA, isEnter? "ENTER_DATA" : "EXIT_DATA", clauses, null);
    pb.setBody(Bcons.emptyBody());
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject clause = arg.getArg();
      if(clause == null) continue; 
      Xobject clauseArg = clause.right();
      String clauseName = clause.left().getName();
      ACCpragma pragma = ACCpragma.valueOf(clauseName);
      ACCpragma newClause = null;
      switch(pragma){
      case COPY:
        newClause = isEnter? ACCpragma.COPYIN : ACCpragma.COPYOUT;
        break;
      case CREATE:
        newClause = isEnter? ACCpragma.CREATE : ACCpragma.DELETE;
        break;
      case COPYIN:
        newClause = isEnter? ACCpragma.COPYIN : ACCpragma.DELETE;
        break;
      case COPYOUT:
        newClause = isEnter? ACCpragma.CREATE : ACCpragma.COPYOUT;
        break;
      default:
        
      }
      if(newClause != null){
                arg.setArg(Xcons.List(Xcons.String(newClause.toString()), clauseArg));
      }
    }
    pb.replace(newPB);
  }

  private XMPdevice getXACCdevice(XobjList clauses, Block block){
    XMPdevice onDevice = null;
    
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject x = arg.getArg();
      if(x == null) continue;
      String clauseName = x.left().getString();
      ACCpragma accClause = ACCpragma.valueOf(clauseName);
      //if(accClause == null) continue;
      if(accClause == ACCpragma.ON_DEVICE){
        String deviceName = x.right().getString();
        onDevice = (XMPdevice)_globalDecl.getXMPobject(deviceName, block);
        if (onDevice == null) XMP.error(x.getLineNo(), "wrong device in on_device");
        arg.setArg(null);
      }
    }
    return onDevice;
  }
  
  private XMPlayout getXACClayout(XobjList clauses){
    XMPlayout layout = null;
    Xobject shadow = null;
    
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject x = arg.getArg();
      if(x == null) continue;
      String clauseName = x.left().getString();
      ACCpragma accClause = ACCpragma.valueOf(clauseName);
      if(accClause == ACCpragma.LAYOUT){
        layout = new XMPlayout((XobjList)x.right());
        arg.setArg(null);
      }else if(accClause == ACCpragma.SHADOW){
        shadow = (XobjList)x.right();
        arg.setArg(null);
      }
    }
    
    if(layout != null && shadow != null){
      layout.setShadow(shadow);
    }
    
    return layout;
  }
  
  private XMPon getXACCon(XobjList clauses, Block b){
    //XMPlayout layout = null;
    XMPon on = null;
    
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject x = arg.getArg();
      if(x == null) continue;
      String clauseName = x.left().getString();
      ACCpragma accClause = ACCpragma.valueOf(clauseName);
      if(accClause == ACCpragma.ON){
        //layout = new XMPlayout((XobjList)x.right());
        XobjList clauseArg = (XobjList)x.right();
        XobjList array = (XobjList)clauseArg.getArg(0);
        XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(array.getArg(0).getSym());
        if(alignedArray == null){
          XMP.fatal("no aligned array");
        }
        XACCdeviceArray deviceArray = _globalDecl.getXACCdeviceArray(array.getArg(0).getSym(), b);
        on = new XMPon(array, deviceArray);
        arg.setArg(null);
      }
    }
    
    return on;
  }
  
  /*
   * rewrite ACC clauses
   */
  private XMPdevice rewriteACCClauses(Xobject expr, PragmaBlock pragmaBlock,
                                      BlockList body, BlockList devLoopBody, Ident devLoopId, XMPdevice onDevice, XMPlayout layout){

//    XMPdevice onDevice = null;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);

    for (iter.init(); !iter.end(); iter.next()){
      Xobject x = iter.getXobject();
      if (x == null) continue;

      if (x.Opcode() == Xcode.LIST){
        if (x.left() == null || x.left().Opcode() != Xcode.STRING) continue;
        
        String clauseName = x.left().getString();
        ACCpragma accClause = ACCpragma.valueOf(clauseName); 
        if(accClause != null){
          switch(accClause){
          case HOST:
          case DEVICE:
          case USE_DEVICE:
          case PRIVATE:
          case FIRSTPRIVATE:   
          case DEVICE_RESIDENT:
            break;
          default:
            if(!accClause.isDataClause()) continue;
          }
          
          XobjList itemList  = (XobjList)x.right();
          for(int i = 0; i < itemList.Nargs(); i++){
            Xobject item = itemList.getArg(i);
            if(item.Opcode() == Xcode.VAR){
              //item is variable or arrayAddr
              try{
                itemList.setArg(i, rewriteACCArrayAddr(item, pragmaBlock.getParentBlock()/*not correct*/, body, devLoopBody, devLoopId, onDevice, layout));
              }catch (XMPexception e){
                XMP.error(x.getLineNo(), e.getMessage());
              }
            }else if(item.Opcode() == Xcode.LIST){
              //item is arrayRef
              try{
                itemList.setArg(i, rewriteACCArrayRef(item, pragmaBlock, body));
              }catch(XMPexception e){
                XMP.error(x.getLineNo(), e.getMessage());
              }
            }
          }
        }

        if (x.left().getString().equals("PRIVATE")){
          if (!pragmaBlock.getPragma().contains("LOOP")) continue;

          XobjList itemList = (XobjList)x.right();

          // find loop variable
          Xobject loop_var = null;
          BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
          for (Block b = pragmaBlock.getBody().getHead();
              b != null;
              b = b.getNext()){
            if (b.Opcode() == Xcode.F_DO_STATEMENT){
              loop_var = ((FdoBlock)b).getInductionVar();
            }
          }
          if (loop_var == null) continue;

          // check if the clause has contained the loop variable
          boolean flag = false;
          Iterator<Xobject> j = itemList.iterator();
          while (j.hasNext()){
            Xobject item = j.next();
            if (item.getName().equals(loop_var.getName())){
              flag = true;
            }
          }

          // add the loop variable to the clause
          if (!flag){
            itemList.add(loop_var);
            }
        }
      } 
      }
    return null;
  }
  
  private Xobject rewriteACCArrayAddr(Xobject arrayAddr, Block block, BlockList body, BlockList deviceLoopBody,
        Ident deviceLoopCounter, XMPdevice onDevice, XMPlayout layout) throws XMPexception {
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), block);
      XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), block);

      if (alignedArray == null && coarray == null) {
          return arrayAddr;
      }
      else if(alignedArray != null && coarray == null){ // only alignedArray
          if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
                  alignedArray.isParameter()){
              Xobject arrayAddrRef = alignedArray.getAddrId().Ref();
              Ident descId = alignedArray.getDescId();
              XobjList arrayRef;
              
              if(deviceLoopCounter == null){
              String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
              Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);

              Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
              body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
              
              arrayRef = Xcons.List(arrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
              }else{

                /*              
                _XACC_init_device_array(_XMP_DESC_a, _XMP_DESC_d);
                _XACC_split_device_array_BLOCK(_XMP_DESC_a, 0);
                _XACC_calc_size(_XMP_DESC_a);
                */
                Ident layoutedArrayDescId = null;
                if(layout != null){
                  layoutedArrayDescId = body.declLocalIdent(XACC_DESC_PREFIX + arrayAddr.getSym(), Xtype.voidPtrType);
                  Block initDeviceArrayFuncCall = _globalDecl.createFuncCallBlock("_XACC_init_layouted_array", Xcons.List(layoutedArrayDescId.getAddr(), descId.Ref(), onDevice.getDescId().Ref()));
                  body.add(initDeviceArrayFuncCall);
                  for(int dim = alignedArray.getDim() - 1; dim >= 0; dim--){
                    int distManner = layout.getDistMannerAt(dim);
                    //if(distManner == XMPlayout.DUPLICATION) continue;
                    String mannerStr = XMPlayout.getDistMannerString(distManner);
                    Block splitDeviceArrayBlockCall = _globalDecl.createFuncCallBlock("_XACC_split_layouted_array_" + mannerStr, Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim)));
                    body.add(splitDeviceArrayBlockCall);
					if(! layout.hasShadow()) continue;
                    int shadowType = layout.getShadowTypeAt(dim);
                    if(shadowType != XMPlayout.SHADOW_NONE){
                      String shadowTypeStr = XMPlayout.getShadowTypeString(shadowType);

                      Block setShadowFuncCall = _globalDecl.createFuncCallBlock("_XACC_set_shadow_" + shadowTypeStr, 
                          Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim), layout.getShadowLoAt(dim), layout.getShadowHiAt(dim)));
                      body.add(setShadowFuncCall);
                    }
                  }
                  Block calcDeviceArraySizeCall = _globalDecl.createFuncCallBlock("_XACC_calc_size", Xcons.List(layoutedArrayDescId.Ref()));
                  body.add(calcDeviceArraySizeCall);
                  //alignedArray.setLayout(layout);


                  XACCdeviceArray deviceArray = new XACCdeviceArray(layoutedArrayDescId, alignedArray, layout); 
                  if(block != null){
                    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(block);
                    localXMPsymbolTable.putXACCdeviceArray(deviceArray);
                  }else{
                    _globalDecl.putXACCdeviceArray(deviceArray);
                  }
                }
                if(layoutedArrayDescId == null){
                  XACCdeviceArray layoutedArray = _globalDecl.getXACCdeviceArray(arrayAddr.getSym(), block);
                  layoutedArrayDescId = layoutedArray.getDescId();
                }
                
                String arraySizeName = "_XACC_size_" + arrayAddr.getSym();
                String arrayOffsetName = "_XACC_offset_" + arrayAddr.getSym();
                Ident arraySizeId = deviceLoopBody.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
                Ident arrayOffsetId = deviceLoopBody.declLocalIdent(arrayOffsetName, Xtype.unsignedlonglongType);
                Block getRangeFuncCall = _globalDecl.createFuncCallBlock("_XACC_get_size", Xcons.List(layoutedArrayDescId.Ref(), arrayOffsetId.getAddr(), arraySizeId.getAddr(), deviceLoopCounter.Ref()));
                deviceLoopBody.add(getRangeFuncCall);
                arrayRef = Xcons.List(arrayAddrRef, Xcons.List(arrayOffsetId.Ref(), arraySizeId.Ref()));
              }
              
              arrayRef.setIsRewrittedByXmp(true);
              return arrayRef;
          }
          else {
              return arrayAddr;
          }
      } else if(alignedArray == null && coarray != null){  // only coarray
          Xobject coarrayAddrRef = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + arrayAddr.getSym()).Ref();
          Ident descId = coarray.getDescId();
          
          String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
          Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
          
          Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
          body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
          
          XobjList arrayRef = Xcons.List(coarrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
          
          arrayRef.setIsRewrittedByXmp(true);
          return arrayRef;
      } else{ // no execute
          return arrayAddr;
      }
  }
  
  private Xobject rewriteACCArrayRef(Xobject arrayRef, Block block, BlockList body) throws XMPexception {
      Xobject arrayAddr = arrayRef.getArg(0);
      
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), block);
      XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), block);
      
      if(alignedArray != null && coarray == null){ //only alignedArray
          Xobject alignedArrayAddrRef = alignedArray.getAddrId().Ref();
          arrayRef.setArg(0, alignedArrayAddrRef);
      }else if(alignedArray == null && coarray != null){ //only coarray
          Xobject coarrayAddrRef = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + arrayAddr.getSym()).Ref();
          arrayRef.setArg(0, coarrayAddrRef);
      }
      return arrayRef;
  }
}
