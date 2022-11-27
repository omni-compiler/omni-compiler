/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import exc.block.FuncDefBlock;
import exc.block.FunctionBlock;
import xcodeml.util.XmOption;

import exc.object.*;
import exc.openacc.*;

public class OMPtoAccTranslator extends AccTranslator {
  public OMPtoAccTranslator(XobjectFile xobjFile){
    super(xobjFile);

    // _infoReader = new AccInfoReader(_globalDecl);
    _infoReader = new OMPtoAccInfoReader(_globalDecl);
  }
}
