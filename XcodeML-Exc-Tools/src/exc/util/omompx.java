/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import exc.object.XobjectFile;

import exc.openmp.OMP;
import exc.openmp.OMPtranslate;

import exc.xcalablemp.XMP;
import exc.xcalablemp.XMPglobalDecl;
import exc.xcalablemp.XMPtranslate;
import exc.xcalablemp.XMPrealloc;
import exc.xcodeml.XcodeMLtools;
import exc.xcodeml.XcodeMLtools_F;
import exc.xcodeml.XcodeMLtools_C;

import xcodeml.XmLanguage;
import xcodeml.XmObj;
import xcodeml.binding.XmXcodeProgram;
import xcodeml.util.*;


// For removing Relaxer
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
import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory;
import xcodeml.XmException;


/**
 * OpenMP-supported XcodeML to XcodeML translator
 */
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
      "arguments: <-xc|-xf> <-l> <-fopenmp> <-dxcode> <-ddecomp> <-dump>",
      "           <input XcodeML file>",
      "           <-o output reconstructed XcodeML file>",
      "",
      "  -xc          process XcodeML/C document.",
      "  -xf          process XcodeML/Fortran document.",
      "  -l           suppress line directive in decompiled code.",
      "  -i           output indented XcodeML.",
      "  -fopenmp     enable OpenMP translation.",
      "  -fatomicio   enable transforming Fortran IO statements to atomic operations.",
      "  -w N         set max columns to N for Fortran source.",
      "  -gnu         decompile for GNU Fortran (default).",
      "  -intel       decompile for Intel Fortran.",
      "  -decomp      output decompiled source code.",
      "",
      " Debug Options:",
      "  -d           enable output debug message.",
      "  -dxcode      output Xcode file as <input file>.x",
      "  -dump        output Xcode file and decompiled file to standard output.",
      "  -domp        enable output OpenMP translation debug message.",
      " Profiling Options:",
      "  -profile         Emit XMP directive profiling code only for specified directives.",
      "  -allprofile      Emit XMP directive profiling code for all directives.",
      "  -with-scalasca   Emit Scalasca instrumentation.",
      "  -with-tlog       Emit tlog insturumentation.",
      "",
      "  -enable-threads	enable 'threads' clause",
      "  -enable-gpu	enable xmp-dev directive/clauses"
    };
        
    for(String line : lines) {
      System.err.println(line);
    }
    System.exit(1);
  }
    
  public static void main(String[] args) throws Exception
  {
    String inXmlFile = null;
    String outXmlFile = null;
    String lang = "C";
    boolean openMP = false;
    boolean xcalableMP = false;
    boolean xcalableMPthreads = false;
    boolean xcalableMPGPU = false;
    boolean xmpf = false;
    boolean outputXcode = false;
    boolean outputDecomp = false;
    boolean dump = false;
    boolean indent = false;
    boolean all_profile = false;
    boolean selective_profile = false;
    boolean doScalasca = false;
    boolean doTlog = false;
    int maxColumns = 0;
    boolean useRelaxerTranslatorInput = false;
    boolean useRelaxerTranslatorOutput = false;
    boolean useRelaxerDecompilerIutput = false;
        
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
      } else if(arg.equals("-i")) {
	indent = true;
      } else if(arg.equals("-fopenmp")) {
	openMP = true;
      } else if(arg.equals("-fxmp")) {
	xcalableMP = true;
      } else if(arg.equals("-enable-threads")) {
	openMP = true;
	xcalableMPthreads = true;
      } else if(arg.equals("-enable-gpu")) {
	xcalableMPGPU = true;
      } else if(arg.equals("-fxmpf")) {
	xmpf = true;
      } else if(arg.equals("-w")) {
	if(narg == null)
	  error("needs argument after -w");
	maxColumns = Integer.parseInt(narg);
	++i;
      } else if(arg.equals("-dxcode")) {
	outputXcode = true;
      } else if(arg.equals("-decomp")) {
	outputDecomp = true;
      } else if(arg.equals("-dump")) {
	dump = true;
	indent = true;
	outputXcode = true;
	outputDecomp = true;
      } else if(arg.equals("-d")) {
	XmOption.setDebugOutput(true);
      } else if(arg.equals("-fatomicio")) {
	XmOption.setIsAtomicIO(true);
      } else if(arg.equals("-domp")) {
	OMP.debugFlag = true;
      } else if(arg.equals("-dxmp")) {
	XMP.debugFlag = true;
      } else if(arg.equals("-o")) {
	if(narg == null)
	  error("needs argument after -o");
	outXmlFile = narg;
	++i;
      } else if(arg.equals("-gnu")) {
	XmOption.setCompilerVendor(XmOption.COMP_VENDOR_GNU);
      } else if(arg.equals("-intel")) {
	XmOption.setCompilerVendor(XmOption.COMP_VENDOR_INTEL);
      } else if(arg.equals("-allprofile")) {
	all_profile = true;
      } else if(arg.equals("-profile")) {
	selective_profile = true;
      } else if (arg.equals("-with-scalasca")) {
	doScalasca = true;
      } else if (arg.equals("-with-tlog")) {
	doTlog = true;
      } else if (arg.equals("-use-relaxer")) {
	useRelaxerTranslatorInput = true;
	useRelaxerTranslatorOutput = true;
	useRelaxerDecompilerIutput = true;
      } else if (arg.equals("-use-relaxer-tin")) {
	useRelaxerTranslatorInput = true;
      } else if (arg.equals("-use-relaxer-tout")) {
	useRelaxerTranslatorOutput = true;
      } else if (arg.equals("-use-relaxer-din")) {
	useRelaxerDecompilerIutput = true;
      } else if(arg.startsWith("-")){
	error("unknown option " + arg);
      } else if(inXmlFile == null) {
	inXmlFile = arg;
      } else {
	error("too many arguments");
      }
    }
        
    if (all_profile == true || selective_profile == true) {
      if (doScalasca == false && doTlog == false) {
	doScalasca = true;
      }
    }

    Reader reader = null;
    Writer xmlWriter = null;
    Writer xcodeWriter = null;
    Writer decompWriter = null;
    File dir = null;
        
    if(inXmlFile == null) {
      reader = new InputStreamReader(System.in);
    } else {
      reader = new BufferedReader(new FileReader(inXmlFile));
      dir = new File(inXmlFile).getParentFile();
    }
        
    if(outXmlFile == null) {
      xmlWriter = new OutputStreamWriter(System.out);
    } else {
      xmlWriter = new BufferedWriter(new FileWriter(outXmlFile));
    }
    
    if(dump || outputXcode) {
      if(dump) {
	xcodeWriter = new OutputStreamWriter(System.out);
      } else {
	xcodeWriter = new BufferedWriter(new FileWriter(inXmlFile + ".x"));
      }
    }
    
    XmToolFactory toolFactory = new XmToolFactory(lang);
    XmOption.setLanguage(XmLanguage.valueOf(lang));
    XmOption.setIsOpenMP(openMP);
    XmOption.setIsXcalableMP(xcalableMP);
    XmOption.setIsXcalableMPthreads(xcalableMPthreads);
    XmOption.setIsXcalableMPGPU(xcalableMPGPU);
    XmOption.setTlogMPIisEnable(doTlog);

    XobjectFile xobjFile;
    String srcPath = inXmlFile;

    if (useRelaxerTranslatorInput) {
      if (XmOption.getLanguage() == XmLanguage.F) {
	// read XcodeML/Fortran
	XcodeMLtools_F tools = new XcodeMLtools_F();
	xobjFile = tools.read(reader);
	if (inXmlFile != null) {
	  reader.close();
	}
      } else {
	// read XcodeML/C
	List<String> readErrorList = new ArrayList<String>();
	XmXcodeProgram xmProg = toolFactory.createXcodeProgram();
	XmValidator validator = toolFactory.createValidator();

	if(!validator.read(reader, xmProg, readErrorList)) {
	  for (String error : readErrorList) {
	    System.err.println(error);
	    System.exit(1);
	  }
	}

	if(inXmlFile != null) {
	  reader.close();
	}

	srcPath = xmProg.getSource();

	// translate XcodeML to Xcode
	XmXmObjToXobjectTranslator xm2xc_translator = toolFactory.createXmObjToXobjectTranslator();
	xobjFile = (XobjectFile)xm2xc_translator.translate((XmObj)xmProg);
	xmProg = null;
      }
    } else { // useRelaxer
      XcodeMLtools tools = null;
      if (XmOption.getLanguage() == XmLanguage.F) {
	tools = new XcodeMLtools_F();
      } else {
	tools = new XcodeMLtools_C();
      }
      // read XcodeML
      xobjFile = tools.read(reader);
      if (inXmlFile != null) {
	reader.close();
      }
    }
        
    String baseName = null;
    if(dump || srcPath == null || srcPath.indexOf("<") >= 0 ) {
      srcPath = null;
    } else {
      String fileName = new File(srcPath).getName();
      int idx = fileName.indexOf(".");
      if(idx < 0) {
	XmLog.fatal("invalid source file name : " + fileName);
      }
      baseName = fileName.substring(0, idx);
    }
        
    if(xobjFile == null)
      System.exit(1);
        
    // Output Xcode
    if(xcodeWriter != null) {
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }
        
    System.gc();
        
    // XcalableMP translation
    if(xcalableMP) {
      XMPglobalDecl globalDecl = new XMPglobalDecl(xobjFile);
      XMPtranslate xmpTranslator = new XMPtranslate(globalDecl);
      XMPrealloc xmpReallocator = new XMPrealloc(globalDecl);

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
      xobjFile.addHeaderLine("include \"xmp_func_decl.h\"");
      xobjFile.addHeaderLine("include \"xmp_index_macro.h\"");
      xobjFile.addHeaderLine("include \"xmp_comm_macro.h\"");
      if(all_profile || selective_profile){
	if (doScalasca == true) {
	  xobjFile.addHeaderLine("include \"xmp_scalasca.h\"");
	}else if (doTlog == true) {
	  xobjFile.addHeaderLine("include \"xmp_tlog.h\"");
	}
      }
      xmpTranslator.finalize();

      if(xcodeWriter != null) {
	xobjFile.Output(xcodeWriter);
	xcodeWriter.flush();
      }
    }

    if(xmpf){// XcalableMP xmpF translation
      exc.xmpF.XMPtranslate xmp_translator = 
	new exc.xmpF.XMPtranslate(xobjFile);
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
        
    if(!dump && outputXcode) {
      xcodeWriter.close();
    }
        
    // translate Xcode to XcodeML
    XmXobjectToXmObjTranslator xc2xm_translator = toolFactory.createXobjectToXmObjTranslator();
    XmXcodeProgram xmprog = null;
    Document xcodeDoc = null;
    if (useRelaxerTranslatorOutput) {
      xmprog = (XmXcodeProgram)xc2xm_translator.translate(xobjFile);
      xobjFile = null;
      // Output XcodeML
      if(indent) {
	StringBuffer buf = new StringBuffer(1024 * 1024);
	xmprog.makeTextElement(buf);
	if(!dump && !outputDecomp) {
	  xmprog = null;
	}
	StringReader xmlReader = new StringReader(buf.toString());
	buf = null;
	XmUtil.transformToIndentedXml(2, xmlReader, xmlWriter);
	xmlReader.close();
      } else {
	xmprog.makeTextElement(xmlWriter);
	if(!dump && !outputDecomp) {
	  xmprog = null;
	}
      }
    } else { // useRelaxer
      XmXobjectToXcodeTranslator xc2xcodeTranslator = null;
      if (lang.equals("F")) {
	xc2xcodeTranslator = new XmfXobjectToXcodeTranslator();
      } else {
	xc2xcodeTranslator = new XmcXobjectToXcodeTranslator();
      }

      xcodeDoc = xc2xcodeTranslator.write(xobjFile);

      Transformer transformer = null;
      try {
	transformer = TransformerFactory.newInstance().newTransformer();
      } catch(TransformerConfigurationException e) {
	throw new XmException(e);
      }

      transformer.setOutputProperty(OutputKeys.METHOD, "xml");

      if (indent) {
	final int indentSpaces = 2;
	transformer.setOutputProperty(OutputKeys.INDENT, "yes");
	transformer.setOutputProperty(
				      OutputPropertiesFactory.S_KEY_INDENT_AMOUNT, "" + indentSpaces);
      }
      try {
	transformer.transform(new DOMSource(xcodeDoc),
			      new StreamResult(xmlWriter));
      } catch(TransformerException e) {
	throw new XmException(e);
      }

      if (!dump && !outputDecomp) {
	xmprog = null;
      } else {
	// read XcodeML/C again. Make xmprog.
	if (outXmlFile != null && useRelaxerDecompilerIutput) {
	  reader = new BufferedReader(new FileReader(outXmlFile));
	  List<String> readErrorList = new ArrayList<String>();
	  xmprog = toolFactory.createXcodeProgram();
	  XmValidator validator = toolFactory.createValidator();
	  if (!validator.read(reader, xmprog, readErrorList)) {
	    for (String error : readErrorList) {
	      System.err.println(error);
	      System.exit(1);
	    }
	  }
	  reader.close();
	  xcodeDoc = null;
	}
      }
    }
        
    xmlWriter.flush();
        
    if(outXmlFile != null) {
      xmlWriter.close();
      xmlWriter = null;
    }
        
    // Decompile
    XmDecompilerContext context = null;
    if(lang.equals("F")) {
      context = toolFactory.createDecompilerContext();
      if(maxColumns > 0)
	context.setProperty(XmDecompilerContext.KEY_MAX_COLUMNS, "" + maxColumns);
    }
        
    if(outputDecomp) {
      if(dump || srcPath == null) {
	decompWriter = new OutputStreamWriter(System.out);
      } else {
	// set decompile writer
	String newFileName = baseName + "." + (XmOption.isLanguageC() ? "c" : "F90");
	String newFileName2 = baseName + "." + (XmOption.isLanguageC() ? "c" : "f90");
	File newFile = new File(dir, newFileName);
	File newFile2 = new File(dir, newFileName2);
                
	if(newFile.exists())
	  newFile.renameTo(new File(dir, newFileName + ".i"));
	if(newFile2.exists())
	  newFile2.renameTo(new File(dir, newFileName2 + ".i"));
                
	decompWriter = new BufferedWriter(new FileWriter(newFile));
      }
            
      XmDecompiler decompiler = toolFactory.createDecompiler();
      if (useRelaxerDecompilerIutput) {
	decompiler.decompile(context, xmprog, decompWriter);
      } else {
	if (xcodeDoc == null) {
	  javax.xml.parsers.DocumentBuilderFactory docFactory = javax.xml.parsers.DocumentBuilderFactory.newInstance();
	  javax.xml.parsers.DocumentBuilder builder = docFactory.newDocumentBuilder();
	  xcodeDoc = builder.parse(outXmlFile);
	}
	decompiler.decompile(context, xcodeDoc, decompWriter);
      }
      decompWriter.flush();
    
      if(!dump && outputDecomp) {
	decompWriter.close();
      }
    }
  }
}
