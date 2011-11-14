/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import xcodeml.util.*;

import org.w3c.dom.*;
import javax.xml.xpath.*;
import java.util.Set;
import java.util.List;
import java.util.ArrayList;

/**
 * Collector of variable references.
 */
public class XmcVarCollector {
    public XmcVarCollector(Set<String> variables) {
        this.variables = variables;

        XPathFactory xpathFactory = XPathFactory.newInstance();
        XPath xpath = xpathFactory.newXPath();

        String[] xpathExprStrings = {
            "descendant::Var/text()",
            "descendant::varAddr/text()",
            "descendant::funcAddr/text()",
            "descendant::memberRef/text()",
        };

        xpathExprs = new ArrayList<XPathExpression>();
        try {
            for (String xpathExprStr : xpathExprStrings) {
                xpathExprs.add(xpath.compile(xpathExprStr));
            }
        } catch (XPathExpressionException e) {
            throw new XmTranslationException(e);
        }
    }

    public void collectVars(Node node) {
        try {
            for (XPathExpression expr : xpathExprs) {
                Object result = expr.evaluate(node, XPathConstants.NODESET);
                NodeList nodes = (NodeList) result;
                for (int i = 0; i < nodes.getLength(); i++) {
                    variables.add(nodes.item(i).getNodeValue());
                }
            }
        } catch (XPathExpressionException e) {
            throw new XmTranslationException(e);
        }
    }

    private Set<String> variables;
    private List<XPathExpression> xpathExprs;
}
