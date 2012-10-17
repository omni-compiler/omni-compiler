/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import xcodeml.XmException;
import exc.object.Ident;
import exc.object.XobjList;
import exc.object.Xobject;
import exc.object.XobjectDefEnv;
import exc.object.XobjectFile;
import exc.object.Xtype;


public abstract class XmXobjectToXcodeTranslator {

    protected static final String TRUE_STR = "1";

    public Document write(XobjectFile xobjFile) throws XmException {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            doc = builder.newDocument();

            preprocess(xobjFile);

            Element root =
                createElement("XcodeProgram",
                              "language", xobjFile.getLanguageAttribute(),
                              "version", xobjFile.getVersion(),
                              "compiler-info", xobjFile.getCompilerInfo(),
                              "source", xobjFile.getSourceFileName(),
                              "time", xobjFile.getTime());

            Element typeTableNode = transTypeTable(xobjFile.getTypeList());
            Element globalSymbolsNode =
                transGlobalSymbols(xobjFile.getGlobalIdentList());
            Element globalDeclNode = createElement("globalDeclarations");

            // header lines
            addChildNodes(globalDeclNode,
                          transLines(xobjFile.getHeaderLines()));

            transGlobalDeclarations(globalDeclNode, xobjFile);

            // type table, symbol table, global declarations
            addChildNodes(root,
                          typeTableNode,
                          globalSymbolsNode,
                          globalDeclNode);

            // tailer lines
            addChildNodes(globalDeclNode,
                          transLines(xobjFile.getTailerLines()));

            doc.appendChild(root);

            postprocess(xobjFile);
        } catch (ParserConfigurationException e) {
            throw new XmException(e);
        }
        return doc;
    }

    protected Document doc;

    abstract Node trans(Xobject xobj);

    protected Node trans(String str) {
        return doc.createTextNode(str);
    }

    abstract void transGlobalDeclarations(Element globalDecl, XobjectDefEnv defList);

    abstract protected Element transType(Xtype type);

    abstract protected Element transIdent(Ident ident);

    protected Element transTypeTable(List<Xtype> xtypeList) {
      Element e = createElement("typeTable");
      for(int i = 0; i < xtypeList.size(); i++){
	Xtype xtype = xtypeList.get(i);
	addChildNodes(e, transType(xtype));
      }
//         for (Xtype xtype : xtypeList) {
//             addChildNodes(e, transType(xtype));
//         }
        return e;
    }

    protected Element transGlobalSymbols(Xobject xobj) {
        XobjList identList = (XobjList)xobj;
        Element e = createElement("globalSymbols");
        for (Xobject ident : identList) {
            addChildNodes(e, transIdent((Ident)ident));
        }
        return e;
    }

    protected void preprocess(XobjectFile xobjFile) {
    }

    protected void postprocess(XobjectFile xobjFile) {
    }

    protected Element createElement(String name, String ... attrs) {
        return addAttributes(doc.createElement(name), attrs);
    }

    protected Element addAttributes(Element e, String ... attrs) {
        final int len = attrs.length;
        assert len % 2 == 0;

        for (int i = 0; i < len; i += 2) {
            final String name = attrs[i];
            final String value = attrs[i + 1];
            if (value != null) {
                e.setAttribute(name, value);
            }
        }

        return e;
    }

    protected Element addChildNode(Element e, Node n) {
        if (n != null) {
            e.appendChild(n);
        }
        return e;
    }

    protected Element addChildNodes(Element e, Node ... nodes) {
        for (Node n : nodes) {
            addChildNode(e, n);
        }
        return e;
    }

    protected Node[] transLines(List<String> lines) {
        ArrayList<Node> textList = new ArrayList<Node>(lines != null ? lines.size() : 0);

        if (lines != null) {
            for (String line : lines) {
                if (line == null) {
                    continue;
                }
                String[] splines = line.split("\n");
                for (String spline : splines) {
                    textList.add(addChildNode(createElement("text"),
                                              doc.createTextNode(spline)));
                }
            }
        }
        return textList.toArray(new Node[0]);
    }

    protected String toBoolStr(boolean enabled) {
        return (enabled) ? TRUE_STR : null;
    }

    protected String intFlagToBoolStr(Xobject flag) {
        if (flag == null) {
            return null;
        }
        return (flag.getInt() == 1) ? TRUE_STR : null;
    }
}

