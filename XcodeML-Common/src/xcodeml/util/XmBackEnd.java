/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import xcodeml.XmException;
import xcodeml.binding.XmXcodeProgram;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmDecompiler;
import xcodeml.util.XmOption;
import xcodeml.util.XmToolFactory;
import xcodeml.util.XmValidator;

import org.w3c.dom.Document;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

/**
 * Run XcodeML decompiler.
 */
public class XmBackEnd
{
    private String _commandName;
    
    private BufferedReader _reader;

    private String _inputFilePath;

    private XmToolFactory _toolFactory;
    
    public XmBackEnd(String langId, String commandName) throws XmException
    {
        _commandName = commandName;
        _toolFactory = new XmToolFactory(langId);
    }

    private void _error(String s)
    {
        System.out.flush();
        System.err.flush();
        System.err.println(s);
        System.err.flush();
        System.exit(1);
    }

    private void _error(Exception e, String s)
    {
        System.out.flush();
        System.err.flush();
        System.err.println(s + " : " + e.getMessage());
        if(XmOption.isDebugOutput())
            e.printStackTrace();
        System.err.flush();
        System.exit(1);
    }

    private void _usage()
    {
        final String[] lines = {
            "usage : " + _commandName + " <XML_INPUT_FILE> <-o OUTPUT_FILE>",
            "",
            "  XML_INPUT_FILE ... XcodeML file for input. If not specified, using stdin.",
            "  OUTPUT_FILE    ... Program code file for output. If not specified, using stdout.",
            "",
            "  -x            add input XcodeML to C source file as comments.",
            "  -l            suppress line directive.",
            "  -w N          set max columns to N for Fortran source code.",
            "  -d            enable debug output.",
            "  -h,--help     print this message.",
        };
        
        for(String line : lines) {
            System.err.println(line);
        }
    }

    private boolean _openInputFile()
    {
        if(_reader != null) {
            try {
                _reader.close();
            } catch(IOException e) {
                e.printStackTrace();
                return false;
            }
        }
        try {
            if(_inputFilePath == null) {
                _reader = new BufferedReader(new InputStreamReader(System.in));
            } else {
                _reader = new BufferedReader(new FileReader(_inputFilePath));
            }
            return true;
        } catch(IOException e) {
            _error(e, "Cannot open input file.");
            return false;
        }
    }

    /**
     * runs decompiler and returns exit code.
     */
    public int run(String[] args) throws XmException
    {
        PrintWriter writer = null;
        String outputFilePath = null;
        boolean addXml = false;
        int maxColumns = 0;

        boolean coarray_noUseStmt = false;
        for(int i = 0; i < args.length; ++i) {
            String arg = args[i];
            String narg = (i < args.length - 1) ? args[i + 1] : null;

            if(arg.startsWith("-")) {
                if(arg.equals("-o")) {
                    if(narg == null) {
                        _error("needs argument after -o.");
                    }
                    outputFilePath = narg;
                    ++i;
                } else if(arg.equals("-d")) {
                    XmOption.setDebugOutput(true);
                } else if(arg.equals("-x")) {
                    addXml = true;
                } else if(arg.equals("-l")) {
                    XmOption.setIsSuppressLineDirective(true);
                } else if (arg.equals("-w")) {
                    if (narg == null) {
                        _error("needs argument after -w.");
                    }
                    try {
                        maxColumns = Integer.parseInt(narg);
                        ++i;
                    } catch (NumberFormatException e) {
                        _error("invalid number after -w.");
                    }
                } else if(arg.equals("-fcoarray-no-use-statement")) {       // TEMPORARY
                    coarray_noUseStmt = true;
                } else if(arg.equals("--help") || arg.equals("-h")) {
                    _usage();
                    System.exit(1);
                } else {
                    _error("Unknown option " + arg + ".");
                }
            } else if(_inputFilePath == null) {
                _inputFilePath = arg;
            } else {
                _error("Too many input files.");
            }
        }
        XmOption.setCoarrayNoUseStatement(coarray_noUseStmt);

        if(!_openInputFile())
            return 1;

        try {
            if(outputFilePath == null) {
                // useful for debug
                writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
            } else {
                writer = new PrintWriter(new BufferedWriter(new FileWriter(outputFilePath)));
            }
        } catch(IOException e) {
            _error(e, "Cannot open output file.");
            return 1;
        }

        // XmXcodeProgram xmprog = _toolFactory.createXcodeProgram();
        // XmValidator validator = _toolFactory.createValidator();
        List<String> errorList = new ArrayList<String>();

        try {
            // if(validator.read(_reader, xmprog, errorList) == false) {
            //     for(String error : errorList)
            //         _error("Error at reading XML file: " + error);
            //     return 1;
            // }
            
            if(addXml) {
                if(!_openInputFile())
                    return 1;
                try {
                    String line = null;
                    while((line = _reader.readLine()) != null) {
                        writer.println("//" + line);
                    }
                } catch(IOException e) {
                    _error(e, "Failed to read input file.");
                    return 1;
                }
            }

            XmDecompiler decompiler = _toolFactory.createDecompiler();
            XmDecompilerContext context = _toolFactory.createDecompilerContext();
            
            if(maxColumns > 0) {
                context.setProperty(XmDecompilerContext.KEY_MAX_COLUMNS, "" + maxColumns);
            }

	    try {
	      javax.xml.parsers.DocumentBuilderFactory docFactory = javax.xml.parsers.DocumentBuilderFactory.newInstance();
	      javax.xml.parsers.DocumentBuilder builder = docFactory.newDocumentBuilder();
	      Document xcodeDoc;
	      if (_inputFilePath != null)
		xcodeDoc = builder.parse(_inputFilePath);
	      else
		xcodeDoc = builder.parse(System.in);
	      decompiler.decompile(context, xcodeDoc, writer);
	    } catch (ParserConfigurationException e) {
	      _error(e, "Error at decompiling");
	      return 1;
	    } catch (SAXException e) {
	      _error(e, "Error at decompiling");
	      return 1;
	    } catch (IOException e) {
	      _error(e, "Error at decompiling");
	      return 1;
	    }

            // try {
            //     decompiler.decompile(context, xmprog, writer);
            // } catch(XmBindingException e) {
            //     _error(e, "Error at decompiling");
            //     _error("  location: " + e.getElementDesc());
            //     return 1;
            // } catch(XmException e) {
            //     _error(e, "Error at decompiling");
            //     return 1;
            // }
            writer.flush();

            return 0;

        } finally {
            if(_inputFilePath != null)
                try { _reader.close(); } catch(Exception e) {}
            if(outputFilePath != null)
                try { writer.close(); } catch(Exception e) {}
        }
    }
}
