/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import exc.object.XobjectFile;
import exc.object.IXobject;

import exc.openacc.ACC;
import exc.openacc.AccDevice;
import exc.openacc.AccTranslator;
import exc.openmp.OMP;
import exc.openmp.OMPtranslate;

import exc.xcalablemp.XMP;
import exc.xcalablemp.XMPglobalDecl;
import exc.xcalablemp.XMPtranslate;
import exc.xcalablemp.XMPrealloc;

import exc.xcodeml.XcodeMLtools;
import exc.xcodeml.XcodeMLtools_F;
import exc.xcodeml.XcodeMLtools_Fmod;
import exc.xcodeml.XcodeMLtools_C;

import xcodeml.util.*;

import exc.xcodeml.XmXobjectToXcodeTranslator;
import exc.xcodeml.XmfXobjectToXcodeTranslator;
import exc.xcodeml.XmcXobjectToXcodeTranslator;

import org.w3c.dom.Document;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

public class omompx
{
  private static void error(String s)
  {
    System.err.println(s);
    System.exit(1);
  }
    
  private static void usage()
  {
    final String[] lines = {
      "arguments: [-xc|-xf] [-l] [-fopenmp] [-f[no]coarray] [-dxcode] [-ddecomp] [-dump]",
      "           <input XcodeML file>",
      "           [-o <output reconstructed XcodeML file>]",
      "",
      "  -xc          process XcodeML/C document.",
      "  -xf          process XcodeML/Fortran document.",
      "  -l           suppress line directive in decompiled code.",
      "  -fopenmp     enable OpenMP translation.",
      "  -fcoarry[=suboption]",
      "               enable coarray translation optionally with a suboption.",
      "  -fnocoarry   pass without coarray translation (default for C).",
      "  -fcoarry-no-use-statement",
      "               supress generation of the USE statement for coarray runtime libraries.",
      "  -fatomicio   enable transforming Fortran IO statements to atomic operations.",
      "  -w N         set max columns to N for Fortran source.",
      "  -gnu         decompile for GNU Fortran (default).",
      "  -intel       decompile for Intel Fortran.",
      "  -M dir       specify where to search for .xmod files",
      "  -max_assumed_shape=N  set max number of assumed-shape arrays of a proedure (for Fortran).",
      "  -decomp      output decompiled source code.",
      "  -silent      no output.",
      "  -rename_main=NAME",
      "               rename the main function NAME.",
      "",
      " Debug Options:",
      "  -d           enable output debug message.",
      "  -dxcode      output Xcode file as <input file>.x",
      "  -dump        output Xcode file and decompiled file to standard output.",
      "  -domp        enable output OpenMP translation debug message.",
      " Profiling Options:",
      "  -scalasca-all       : output results in scalasca format for all directives.",
      "  -scalasca-selective : output results in scalasca format for selected directives.",
      "  -tlog-all           : output results in tlog format for all directives.",
      "  -tlog-selective     : output results in tlog format for selected directives.",
      "",
      "  -enable-threads     : enable 'threads' clause",
      "  -enable-gpu         : enable xmp-dev directive/clauses",
      "  -enable-Fonesided   : enable one-sided functions (Only Fortran)"
    };
        
    for(String line : lines) {
      System.err.println(line);
    }
    System.exit(1);
  }
    
  public static void main(String[] args) throws Exception
  {
    String inXmlFile           = null;
    String outXmlFile          = null;
    String lang                = "C";
    boolean openMP             = false;
    boolean openACC            = false;
    boolean coarray            = true;
    boolean xcalableMP         = false;
    boolean xcalableMPthreads  = false;
    boolean xcalableMPGPU      = false;
    boolean xmpf               = false;
    boolean async              = false;
    boolean xcalableACC        = false;
    boolean xmpt               = false;
    boolean outputXcode        = false;
    boolean outputDecomp       = false;
    boolean dump               = false;
    boolean all_profile        = false;
    boolean selective_profile  = false;
    boolean doScalasca         = false;
    boolean doTlog             = false;
    boolean silent             = false;
    boolean Fonesided          = false;
    int maxColumns             = 0;
    String coarray_suboption   = "";        // HIDDEN
    boolean coarray_noUseStmt  = false;     // TEMPORARY
    int accDefaultVectorLength = 0;
    boolean accDisableReadOnlyDataCache = false;
        
    // environment variable analysis
    Boolean xmpf_onlyCafMode = "1".equals(System.getenv("XMP_ONLYCAF"));
    Boolean xmpf_skipCafMode = "1".equals(System.getenv("XMP_SKIPCAF"));

    for(int i = 0; i < args.length; ++i) {
      String arg = args[i];
      String narg = (i < args.length - 1) ? args[i + 1] : null;
    
      if(arg.equals("-h") || arg.equals("--help")) {
        usage();
      } else if(arg.equals("-xc")) {
        lang = "C";
      } else if(arg.equals("-xf")) {
        lang = "F";
      } else if(arg.equals("-l")) {
        XmOption.setIsSuppressLineDirective(true);
      } else if(arg.equals("-fopenmp")) {
        openMP = true;
      } else if(arg.equals("-fcoarray")) {
        coarray = true;
      } else if(arg.equals("-fnocoarray")) {
        coarray = false;
        coarray_noUseStmt = true;
      } else if(arg.startsWith("-fcoarray=")) {                  // HIDDEN
        coarray_suboption += arg.substring(arg.indexOf("=")+1);
      } else if(arg.equals("-fcoarray-no-use-statement")) {       // TEMPORARY
        coarray_noUseStmt = true;
      } else if(arg.equals("-facc")) {
        openACC = true; 
      } else if(arg.equals("-fxmp")) {
        xcalableMP = true;
      } else if(arg.equals("-enable-threads")) {
        openMP = true;
        xcalableMPthreads = true;
      } else if(arg.equals("-enable-gpu")) {
        xcalableMPGPU = true;
      } else if(arg.equals("-enable-Fonesided")) {
        Fonesided = true;
      } else if(arg.equals("-fxmpf")) {
        xmpf = true;
      } else if(arg.equals("-fasync")) {
        async = true;
      } else if(arg.equals("-fxacc")) {
        xcalableACC = true;
      } else if(arg.equals("-fxmpt")) {
	xmpt = true;
      } else if(arg.equals("-w")) {
        if(narg == null)
          error("needs argument after -w");
        maxColumns = Integer.parseInt(narg);
        ++i;
      } else if(arg.equals("-dxcode")) {
        outputXcode = true;
      } else if(arg.equals("-decomp")) {
        outputDecomp = true;
      } else if(arg.equals("-silent")){
        silent = true;
      } else if(arg.startsWith("-rename_main=")) {
        String main_name = arg.substring(arg.indexOf("=") + 1);
        XmOption.setMainName(main_name);
      } else if(arg.equals("-dump")) {
        dump = true;
        outputXcode = true;
        outputDecomp = true;
      } else if(arg.equals("-d")) {
        XmOption.setDebugOutput(true);
      } else if(arg.equals("-fatomicio")) {
        XmOption.setIsAtomicIO(true);
      } else if(arg.equals("-domp")) {
        OMP.debugFlag = true;
      } else if(arg.equals("-dxmp")) {
        exc.xmpF.XMP.debugFlag = true;
      } else if(arg.equals("-o")) {
        if(narg == null)
          error("needs argument after -o");
        outXmlFile = narg;
        ++i;
      } else if(arg.equals("-gnu")) {
        XmOption.setCompilerVendor(XmOption.COMP_VENDOR_GNU);
      } else if(arg.equals("-intel")) {
        XmOption.setCompilerVendor(XmOption.COMP_VENDOR_INTEL);
      } else if (arg.equals("-scalasca-selective")) {
        selective_profile = true;
        doScalasca = true;
      } else if (arg.equals("-scalasca-all")) {
        all_profile = true;
        doScalasca = true;
      } else if (arg.equals("-tlog-selective")) {
        selective_profile = true;
        doTlog = true;
      } else if (arg.equals("-tlog-all")) {
        all_profile = true;
        doTlog = true;
      } else if (arg.startsWith("-M")) { 
          if (arg.equals("-M")) {
            if (narg == null)
              error("needs argument after -M");
            XcodeMLtools_Fmod.addSearchPath(narg);
            ++i;
          }
          else {
            XcodeMLtools_Fmod.addSearchPath(arg.substring(2));
          }
      } else if (arg.startsWith("-max_assumed_shape=")) {
	  String n = arg.substring(19);
	  exc.xmpF.XMP.MAX_ASSUMED_SHAPE = Integer.parseInt(n);
      } else if (arg.equals("-no-ldg")) {
        accDisableReadOnlyDataCache = true;
      } else if (arg.startsWith("-default-veclen=")) {
        String n = arg.substring("-default-veclen=".length());
        accDefaultVectorLength = Integer.parseInt(n);
      } else if (arg.startsWith("-platform=")){
        String n = arg.substring("-platform=".length());
        ACC.platform = ACC.Platform.valueOf(n);
      } else if (arg.startsWith("-device=")){
        String n = arg.substring("-device=".length());
        ACC.device = AccDevice.getDevice(n);
      } else if(arg.startsWith("-")){
        error("unknown option " + arg);
      } else if(inXmlFile == null) {
        inXmlFile = arg;
      } else {
        error("too many arguments");
      }
    }
        
    doScalasca = (all_profile == true || selective_profile == true) 
      && (doScalasca == false && doTlog == false);

    Reader reader = null;
    File dir      = null;
    if(inXmlFile == null) {
      reader = new InputStreamReader(System.in);
    }
    else {
      reader = new BufferedReader(new FileReader(inXmlFile));
      dir = new File(inXmlFile).getParentFile();
    }

    Writer xcodeWriter = null;
    if(dump || outputXcode) {
      if(dump) {
        xcodeWriter = new OutputStreamWriter(System.out);
      } else {
        xcodeWriter = new BufferedWriter(new FileWriter(inXmlFile + ".x"));
      }
    }
   
    XmOption.setLanguage(XmLanguage.valueOf(lang));
    XmOption.setIsOpenMP(openMP);
    XmOption.setIsCoarray(coarray);
    XmOption.setIsAsync(async);
    XmOption.setIsXcalableMP(xcalableMP);
    XmOption.setXmptIsEnabled(xmpt);
    XmOption.setIsXcalableMPthreads(xcalableMPthreads);
    XmOption.setIsXcalableMPGPU(xcalableMPGPU);
    XmOption.setTlogMPIisEnable(doTlog);
    XmOption.setFonesided(Fonesided);
    XmOption.setCoarrayNoUseStatement(coarray_noUseStmt);   // TEMPORARY
    XmOption.setIsXcalableACC(xcalableACC);
    
    // read XcodeML
    XcodeMLtools tools = 
      (XmOption.getLanguage() == XmLanguage.F)? 
      new XcodeMLtools_F() : new XcodeMLtools_C();
    XobjectFile xobjFile = tools.read(reader);
    
    if (inXmlFile != null) reader.close();
    if (xobjFile == null)  System.exit(1);

    String srcPath  = inXmlFile;
    String baseName = null;
    if(dump || srcPath == null || srcPath.indexOf("<") >= 0 ) {
      srcPath = null;
    }
    else {
      String fileName = new File(srcPath).getName();
      int idx = fileName.lastIndexOf(".");
      if(idx < 0) {
        XmLog.fatal("invalid source file name : " + fileName);
      }
      baseName = fileName.substring(0, idx);
    }

    // Output Xcode
    if(xcodeWriter != null) {
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }
        
    System.gc();
        
    // XcalableMP translation
    if(xcalableMP) {
      XMPglobalDecl globalDecl   = new XMPglobalDecl(xobjFile);
      XMPtranslate xmpTranslator = new XMPtranslate(globalDecl);
      XMPrealloc xmpReallocator  = new XMPrealloc(globalDecl);

      // For profile                                                                            
      if(all_profile){
        xmpTranslator.set_all_profile();
      }
      if(selective_profile){
        xmpTranslator.set_selective_profile();
      }
      xmpTranslator.setScalascaEnabled(doScalasca);
      xmpTranslator.setTlogEnabled(doTlog);
      
      xobjFile.iterateDef(xmpTranslator);
      XMP.exitByError();
      xobjFile.iterateDef(xmpReallocator);
      XMP.exitByError();
      globalDecl.setupGlobalConstructor();
      globalDecl.setupGlobalDestructor();
      XMP.exitByError();
      xobjFile.addHeaderLine("# include \"xmp_func_decl.h\"");
      xobjFile.addHeaderLine("# include \"xmp_index_macro.h\"");
      xobjFile.addHeaderLine("# include \"xmp_comm_macro.h\"");
      if(all_profile || selective_profile){
        if (doScalasca == true) {
          xobjFile.addHeaderLine("# include \"xmp_scalasca.h\"");
        }
        else if (doTlog == true) {
          xobjFile.addHeaderLine("# include \"xmp_tlog.h\"");
        }
      }
      
      if(openACC){
        if(xobjFile.findIdent("acc_init", IXobject.FINDKIND_ANY) == null){
          xobjFile.addHeaderLine("# include \"openacc.h\"");
        }
      }
      xmpTranslator.finish();

      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    }

    if (xmpf && (xmpf_skipCafMode || !XmOption.isCoarray())) {
      System.out.println("<SKIP-CAF MODE> XMP/F Coarray translator is " +
                         "bypassed for " + xobjFile.getSourceFileName() + ".");
    }

    if (xmpf && (!xmpf_skipCafMode && XmOption.isCoarray())) {

      // Coarray Fortran pass#3
      exc.xmpF.XMPtransCoarray caf_translator3 =
        new exc.xmpF.XMPtransCoarray(xobjFile, 3, coarray_suboption,
                                     xmpf_onlyCafMode);
      xobjFile.iterateDef(caf_translator3);

      // Coarray Fortran pass#4
      exc.xmpF.XMPtransCoarray caf_translator4 =
        new exc.xmpF.XMPtransCoarray(xobjFile, 4, coarray_suboption,
                                     xmpf_onlyCafMode);
      xobjFile.iterateDef(caf_translator4);

      // Coarray Fortran pass#1
      exc.xmpF.XMPtransCoarray caf_translator1 =
        new exc.xmpF.XMPtransCoarray(xobjFile, 1, coarray_suboption,
                                     xmpf_onlyCafMode);
      xobjFile.iterateDef(caf_translator1);
      if(exc.xmpF.XMP.hasErrors())
        System.exit(1);
      caf_translator1.finish();

      // Coarray Fortran pass#2
      exc.xmpF.XMPtransCoarray caf_translator2 =
        new exc.xmpF.XMPtransCoarray(xobjFile, 2, coarray_suboption,
                                     xmpf_onlyCafMode);
      xobjFile.iterateDef(caf_translator2);
      
      if(exc.xmpF.XMP.hasErrors()) System.exit(1);
      caf_translator2.finish();

      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    }

    if (xmpf && xmpf_onlyCafMode) {
      System.out.println("<ONLY-CAF MODE> XMP/F gloval-view translator is " +
                         "bypassed for " + xobjFile.getSourceFileName() + ".");
    }

    if (xmpf && !xmpf_onlyCafMode) {
      // XMP Fortran
      exc.xmpF.XMPtranslate xmp_translator = new exc.xmpF.XMPtranslate(xobjFile);
      xobjFile.iterateDef(xmp_translator);
      
      if(exc.xmpF.XMP.hasErrors())
        System.exit(1);
      
      xmp_translator.finish();

      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    }
    
    // OpenMP translation
    if(openMP) {
      OMPtranslate omp_translator = new OMPtranslate(xobjFile);
      xobjFile.iterateDef(omp_translator);
            
      if(OMP.hasErrors())
        System.exit(1);
            
      omp_translator.finish();
            
      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    }
    
    if(openACC){
      if(ACC.device == AccDevice.NONE){
        switch(ACC.platform){
          case CUDA:
          case OpenCL:
            ACC.device = AccDevice.getDevice("Fermi");
            break;
        }
      }

      if(accDefaultVectorLength > 0) {
        ACC.device.setDefaultVectorLength(accDefaultVectorLength);
      }

      if(accDisableReadOnlyDataCache == true){
        ACC.device.setUseReadOnlyDataCache(false);
      }

      //XmOption.setDebugOutput(true);
      AccTranslator accTranslator = new AccTranslator(xobjFile, false);
      xobjFile.iterateDef(accTranslator);

      accTranslator.finish();
      
      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    }
    
    if(!dump && outputXcode) {
      xcodeWriter.close();
    }

    // translate Xcode to XcodeML
    // create transformer from Xobject to XcodeML DOM.
    XmXobjectToXcodeTranslator xc2xcodeTranslator = null;
    if (lang.equals("F"))
      xc2xcodeTranslator = new XmfXobjectToXcodeTranslator();
    else
      xc2xcodeTranslator = new XmcXobjectToXcodeTranslator();
    
    Document xcodeDoc = xc2xcodeTranslator.write(xobjFile);

    // transformation from DOM to the file. It means to output DOM to the file.
    if(silent == false){
      try {
        Transformer transformer = TransformerFactory.newInstance().newTransformer();
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        Writer xmlWriter = null;
        if (outXmlFile == null)
          xmlWriter = new OutputStreamWriter(System.out);
        else
          xmlWriter = new BufferedWriter(new FileWriter(outXmlFile));
        
        transformer.transform(new DOMSource(xcodeDoc), new StreamResult(xmlWriter));
        xmlWriter.flush();
        
        if(outXmlFile != null) {
          xmlWriter.close();
          xmlWriter = null;
        }
      } catch(TransformerException e) {
        throw new XmException(e);
      }
    }
    
    // Decompile
    XmDecompilerContext context = null;
    XmToolFactory toolFactory = new XmToolFactory(lang);
    if(lang.equals("F")) {
      context = toolFactory.createDecompilerContext();
      if(maxColumns > 0)
        context.setProperty(XmDecompilerContext.KEY_MAX_COLUMNS, "" + maxColumns);
    }
        
    if(outputDecomp) {
      Writer decompWriter = null;
      if(dump || srcPath == null) {
        decompWriter = new OutputStreamWriter(System.out);
      } 
      else { // set decompile writer
        String newFileName = baseName + "." + (XmOption.isLanguageC() ? "c" : "F90");
        File newFile = new File(dir, newFileName);
                
        if(newFile.exists())
          newFile.renameTo(new File(dir, newFileName + ".i"));
                
        decompWriter = new BufferedWriter(new FileWriter(newFile));
      }

      if (xcodeDoc == null) {
        javax.xml.parsers.DocumentBuilderFactory docFactory = 
          javax.xml.parsers.DocumentBuilderFactory.newInstance();
        javax.xml.parsers.DocumentBuilder builder = docFactory.newDocumentBuilder();
        xcodeDoc = builder.parse(outXmlFile);
      }
      
      XmDecompiler decompiler = toolFactory.createDecompiler();
      decompiler.decompile(context, xcodeDoc, decompWriter);
      decompWriter.flush();
    
      if(!dump && outputDecomp) {
        decompWriter.close();
      }
    }
  }
}
